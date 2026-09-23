"""对外 API：``audit`` 出体检报告，``clean`` 边修边记录。

设计上刻意把「检查」和「修复」分开：检查只读，修复每一步都写进 changes，
所以任何一次清洗都可以回看「到底改了什么」。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from . import __version__
from .checks import Finding, Options, audit_panel
from .table import Panel, read_csv, write_csv

SEVERITY_WEIGHT = {"error": 8.0, "warn": 2.0, "info": 0.5}

Source = Union[str, Panel, Dict[str, Panel]]


def quality_score(findings: Sequence[Finding], n_rows: int) -> float:
    """0~100 的质量分：按严重度加权，再按行数归一。"""
    if n_rows <= 0:
        return 0.0
    penalty = 0.0
    for finding in findings:
        penalty += SEVERITY_WEIGHT.get(finding.severity, 1.0) * min(finding.count, n_rows)
    return float(max(0.0, min(100.0, 100.0 - 100.0 * penalty / n_rows)))


def _as_panels(source: Source) -> Dict[str, Panel]:
    if isinstance(source, Panel):
        return {source.symbol: source}
    if isinstance(source, dict):
        return dict(source)
    return read_csv(str(source))


def audit(source: Source, *, options: Optional[Options] = None) -> Dict[str, Any]:
    """体检：不修改任何数据，返回结构化报告。"""
    opts = options or Options()
    panels = _as_panels(source)
    symbols: Dict[str, Any] = {}
    total_rows = 0
    total_issues = 0
    weighted_score = 0.0

    for symbol, panel in panels.items():
        findings = audit_panel(panel, opts)
        score = quality_score(findings, panel.n_rows)
        total_rows += panel.n_rows
        total_issues += sum(f.count for f in findings)
        weighted_score += score * panel.n_rows
        symbols[symbol] = {
            "symbol": symbol,
            "rows": panel.n_rows,
            "start": str(panel.dates[0]) if panel.n_rows else None,
            "end": str(panel.dates[-1]) if panel.n_rows else None,
            "score": round(score, 2),
            "issues": sum(f.count for f in findings),
            "findings": [f.to_dict() for f in findings],
        }

    return {
        "tool": f"kairos-clean {__version__}",
        "source": source if isinstance(source, str) else None,
        "symbols": symbols,
        "summary": {
            "n_symbols": len(panels),
            "rows": total_rows,
            "issues": total_issues,
            "score": round(weighted_score / total_rows, 2) if total_rows else 0.0,
            "by_severity": {
                sev: sum(f["count"] for s in symbols.values()
                         for f in s["findings"] if f["severity"] == sev)
                for sev in ("error", "warn", "info")
            },
        },
        "options": opts.__dict__.copy(),
    }


# --------------------------------------------------------------------------- #
# 清洗
# --------------------------------------------------------------------------- #
@dataclass
class CleanResult:
    """清洗结果：干净的面板 + 改动日志 + 前后报告。"""

    panels: Dict[str, Panel]
    paths: Dict[str, str]
    changes: List[Dict[str, Any]] = field(default_factory=list)
    before: Dict[str, Any] = field(default_factory=dict)
    after: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        b = self.before.get("summary", {})
        a = self.after.get("summary", {})
        return {
            "symbols": len(self.panels),
            "rows_before": b.get("rows", 0),
            "rows_after": a.get("rows", 0),
            "issues_before": b.get("issues", 0),
            "issues_after": a.get("issues", 0),
            "score_before": b.get("score", 0.0),
            "score_after": a.get("score", 0.0),
            "changes": len(self.changes),
        }


def _log(changes: List[Dict[str, Any]], symbol: str, kind: str, action: str,
         detail: str = "") -> None:
    if len(changes) < 5000:
        changes.append({"symbol": symbol, "kind": kind, "action": action, "detail": detail})


def _dedupe(panel: Panel, changes: List[Dict[str, Any]]) -> Panel:
    keep = np.ones(panel.n_rows, dtype=bool)
    if panel.n_rows < 2:
        return panel
    same = np.diff(panel.dates).astype("int64") == 0
    # 重复日期保留最后一条：从后往前标记
    for idx in np.flatnonzero(same):
        keep[int(idx)] = False
        _log(changes, panel.symbol, "duplicate_dates", "drop",
             f"{panel.dates[int(idx)]} 的重复行（保留后一条）")
    return panel.slice(np.flatnonzero(keep))


def _drop_bad_prices(panel: Panel, changes: List[Dict[str, Any]]) -> Panel:
    price_cols = [c for c in ("open", "high", "low", "close") if c in panel.columns]
    if not price_cols:
        return panel
    bad = np.zeros(panel.n_rows, dtype=bool)
    for name in price_cols:
        bad |= ~np.isfinite(panel.columns[name]) | (panel.columns[name] <= 0)
    if not bad.any():
        return panel
    for idx in np.flatnonzero(bad):
        _log(changes, panel.symbol, "non_positive_price", "drop", str(panel.dates[int(idx)]))
    return panel.slice(np.flatnonzero(~bad))


def _repair_ohlc(panel: Panel, changes: List[Dict[str, Any]]) -> Panel:
    if not panel.has("open", "high", "low", "close"):
        return panel
    o, h, l, c = (panel.get(k).copy() for k in ("open", "high", "low", "close"))
    high = np.maximum.reduce([h, o, c])
    low = np.minimum.reduce([l, o, c])
    fixed = int(np.sum((high != h) | (low != l)))
    if fixed == 0:
        return panel
    for idx in np.flatnonzero((high != h) | (low != l)):
        _log(changes, panel.symbol, "ohlc_violation", "repair",
             f"{panel.dates[int(idx)]}: high/low 修正为 {high[int(idx)]:.4f}/{low[int(idx)]:.4f}")
    columns = dict(panel.columns)
    columns["high"], columns["low"] = high, low
    return Panel(panel.symbol, panel.dates, columns, panel.skipped_rows)


def _fill_gaps(panel: Panel, changes: List[Dict[str, Any]]) -> Panel:
    if panel.n_rows < 2:
        return panel
    days = np.arange(panel.dates[0], panel.dates[-1] + 1, dtype="datetime64[D]")
    days = days[np.is_busday(days)]
    if days.size <= panel.n_rows:
        return panel
    pos = np.searchsorted(days, panel.dates)
    columns = {}
    for name, values in panel.columns.items():
        filled = np.full(days.size, np.nan)
        filled[pos] = values
        # 前向填充
        last = np.nan
        for i in range(days.size):
            if np.isfinite(filled[i]):
                last = filled[i]
            else:
                filled[i] = last
        columns[name] = filled
    if "volume" in columns:                     # 补出来的行没有成交
        missing = np.ones(days.size, dtype=bool)
        missing[pos] = False
        columns["volume"][missing] = 0.0
    added = int(days.size - panel.n_rows)
    _log(changes, panel.symbol, "calendar_gap", "fill", f"补 {added} 个交易日（前向填充）")
    return Panel(panel.symbol, days, columns, panel.skipped_rows)


def _winsorize(panel: Panel, options: Options, changes: List[Dict[str, Any]]) -> Panel:
    if "close" not in panel.columns or panel.n_rows < 20:
        return panel
    close = panel.get("close")
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = np.diff(np.log(close))
    finite = ret[np.isfinite(ret)]
    if finite.size < 20:
        return panel
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    scale = max(1.4826 * mad, 1e-9)
    lower, upper = median - options.outlier_k * scale, median + options.outlier_k * scale
    clipped = np.clip(ret, lower, upper)
    if np.allclose(clipped, ret, equal_nan=True):
        return panel
    new_close = close[0] * np.exp(np.concatenate([[0.0], np.cumsum(clipped)]))
    columns = dict(panel.columns)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(close > 0, new_close / close, 1.0)
    for name in ("open", "high", "low"):        # 同步缩放，保持 OHLC 关系
        if name in columns:
            columns[name] = columns[name] * ratio
    columns["close"] = new_close
    changed = int(np.sum(~np.isclose(clipped, ret, equal_nan=True)))
    _log(changes, panel.symbol, "return_outlier", "winsorize", f"截尾 {changed} 个收益率")
    return Panel(panel.symbol, panel.dates, columns, panel.skipped_rows)


def clean(
    source: Source,
    *,
    out_dir: Optional[str] = None,
    options: Optional[Options] = None,
    drop_duplicates: bool = True,
    repair_ohlc: bool = True,
    fill_gaps: bool = False,
    winsorize: bool = False,
) -> CleanResult:
    """清洗：按顺序去重 → 删坏价 → 修 OHLC →（可选）补行 →（可选）截尾。"""
    opts = options or Options()
    panels = _as_panels(source)
    before = audit(panels, options=opts)
    changes: List[Dict[str, Any]] = []
    cleaned: Dict[str, Panel] = {}

    for symbol, panel in panels.items():
        work = panel
        if drop_duplicates:
            work = _dedupe(work, changes)
        work = _drop_bad_prices(work, changes)
        if repair_ohlc:
            work = _repair_ohlc(work, changes)
        if fill_gaps:
            work = _fill_gaps(work, changes)
        if winsorize:
            work = _winsorize(work, opts, changes)
        cleaned[symbol] = work

    after = audit(cleaned, options=opts)
    paths: Dict[str, str] = {}
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        for symbol, panel in cleaned.items():
            name = symbol if symbol else "data"
            paths[symbol] = write_csv(panel, os.path.join(out_dir, f"{name}.csv"))
    return CleanResult(cleaned, paths, changes, before, after)


__all__ = ["CleanResult", "audit", "clean", "quality_score"]