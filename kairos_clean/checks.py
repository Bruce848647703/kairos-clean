"""体检项：每个函数只看一列/几列，返回一个 Finding。

原则：**只报告、不修改**。修不修、怎么修交给 ``api.clean`` 决定，
这样体检结果可以单独存档、对比。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .table import Panel

#: 拆股/合股的常见比例
SPLIT_RATIOS = (0.5, 1 / 3, 0.2, 0.1, 0.05, 2.0, 3.0, 5.0, 10.0, 20.0)


@dataclass
class Options:
    """体检阈值（都能改）。"""

    stale_days: int = 5
    gap_days: int = 7
    outlier_k: float = 8.0
    jump_threshold: float = 0.25
    split_tolerance: float = 0.03
    max_samples: int = 5

    def __post_init__(self) -> None:
        if self.stale_days < 2:
            raise ValueError("stale_days 至少为 2")
        if self.gap_days < 1:
            raise ValueError("gap_days 至少为 1")
        if self.outlier_k <= 0:
            raise ValueError("outlier_k 必须为正")


@dataclass
class Finding:
    """一条体检结论。"""

    kind: str
    severity: str          # error / warn / info
    count: int
    message: str
    suggestion: str = ""
    samples: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "severity": self.severity, "count": self.count,
                "message": self.message, "suggestion": self.suggestion,
                "samples": self.samples}


def _runs(mask: np.ndarray) -> List[tuple]:
    """连续 True 的区间 ``[start, end)``。"""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(np.diff(padded.astype("int8")))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


def _sample(panel: Panel, index: int, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    row: Dict[str, Any] = {"date": str(panel.dates[index])}
    for name, column in panel.columns.items():
        if index < column.size:
            value = float(column[index])
            row[name] = round(value, 6) if np.isfinite(value) else None
    if extra:
        row.update(extra)
    return row


def check_nan(panel: Panel, opts: Options) -> Optional[Finding]:
    mask = np.zeros(panel.n_rows, dtype=bool)
    for column in panel.columns.values():
        mask |= ~np.isfinite(column)
    if not mask.any():
        return None
    idx = np.flatnonzero(mask)[: opts.max_samples]
    return Finding("missing_values", "warn", int(mask.sum()),
                   f"有 {int(mask.sum())} 个空值/NaN",
                   "按前值填充或直接整行删除（--drop-missing）",
                   [_sample(panel, int(i)) for i in idx])


def check_duplicates(panel: Panel, opts: Options) -> Optional[Finding]:
    dup = np.diff(panel.dates).astype("int64") == 0
    if not dup.any():
        return None
    idx = np.flatnonzero(dup)[: opts.max_samples]
    return Finding("duplicate_dates", "error", int(dup.sum()),
                   f"有 {int(dup.sum())} 个重复日期",
                   "保留最后一条（默认会去重）",
                   [_sample(panel, int(i) + 1) for i in idx])


def check_non_positive(panel: Panel, opts: Options) -> Optional[Finding]:
    price_cols = [c for c in ("open", "high", "low", "close") if c in panel.columns]
    if not price_cols:
        return None
    mask = np.zeros(panel.n_rows, dtype=bool)
    for name in price_cols:
        mask |= panel.columns[name] <= 0
    if not mask.any():
        return None
    idx = np.flatnonzero(mask)[: opts.max_samples]
    return Finding("non_positive_price", "error", int(mask.sum()),
                   f"有 {int(mask.sum())} 行价格 <= 0",
                   "多半是缺省值或负价，建议删除这一行",
                   [_sample(panel, int(i)) for i in idx])


def check_ohlc(panel: Panel, opts: Options) -> Optional[Finding]:
    if not panel.has("open", "high", "low", "close"):
        return None
    o, h, l, c = (panel.get(k) for k in ("open", "high", "low", "close"))
    mask = (h < l) | (h < np.maximum(o, c) - 1e-12) | (l > np.minimum(o, c) + 1e-12)
    mask &= np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c)
    if not mask.any():
        return None
    idx = np.flatnonzero(mask)[: opts.max_samples]
    return Finding("ohlc_violation", "error", int(mask.sum()),
                   f"有 {int(mask.sum())} 行 OHLC 自相矛盾（high/low 不在区间内）",
                   "把 high 抬到 max(o,c)、low 压到 min(o,c) 即可修复（默认修复）",
                   [_sample(panel, int(i)) for i in idx])


def check_gap(panel: Panel, opts: Options) -> Optional[Finding]:
    if panel.n_rows < 2:
        return None
    days = np.diff(panel.dates).astype("int64")
    mask = days > opts.gap_days
    if not mask.any():
        return None
    idx = np.flatnonzero(mask)[: opts.max_samples]
    samples = [{"from": str(panel.dates[int(i)]), "to": str(panel.dates[int(i) + 1]),
                "days": int(days[int(i)])} for i in idx]
    return Finding("calendar_gap", "warn", int(mask.sum()),
                   f"有 {int(mask.sum())} 处相邻日期间隔超过 {opts.gap_days} 天",
                   "可能是停牌/长假/漏行；确认后可前向填充（--fill-gaps）",
                   samples)


def check_stale(panel: Panel, opts: Options) -> Optional[Finding]:
    if "close" not in panel.columns or panel.n_rows < 2:
        return None
    close = panel.get("close")
    same = np.diff(close) == 0
    stale_runs = [r for r in _runs(same) if (r[1] - r[0]) + 1 >= opts.stale_days]
    if not stale_runs:
        return None
    samples = [{"from": str(panel.dates[a]), "to": str(panel.dates[b - 1]),
                "days": int(b - a + 1)} for a, b in stale_runs[: opts.max_samples]]
    return Finding("stale_price", "info", len(stale_runs),
                   f"有 {len(stale_runs)} 段连续 {opts.stale_days} 天以上收盘价不变",
                   "可能是停牌或数据源复用了前值；核对原始行情",
                   samples)


def check_outliers(panel: Panel, opts: Options) -> Optional[Finding]:
    if "close" not in panel.columns or panel.n_rows < 20:
        return None
    close = panel.get("close")
    with np.errstate(divide="ignore", invalid="ignore"):
        log_close = np.log(np.where(close > 0, close, np.nan))
        ret = np.diff(log_close)
    ret = ret[np.isfinite(ret)]
    if ret.size < 20:
        return None
    median = float(np.median(ret))
    mad = float(np.median(np.abs(ret - median)))
    scale = max(1.4826 * mad, 1e-9)
    with np.errstate(divide="ignore", invalid="ignore"):
        full = np.diff(np.log(np.where(close > 0, close, np.nan)))
    z = np.abs((full - median) / scale)
    mask = np.isfinite(z) & (z > opts.outlier_k)
    if not mask.any():
        return None
    idx = np.flatnonzero(mask)[: opts.max_samples]
    samples = [_sample(panel, int(i) + 1, {"z": round(float(z[int(i)]), 1)}) for i in idx]
    return Finding("return_outlier", "warn", int(mask.sum()),
                   f"有 {int(mask.sum())} 个收益率的稳健 z 分数超过 {opts.outlier_k}",
                   "可能是脏数据或真实跳空；确认后可用 --winsorize 截尾",
                   samples)


def check_jumps(panel: Panel, opts: Options) -> Optional[Finding]:
    """单日涨跌超过阈值 —— 重点怀疑未复权的拆股/合股。"""
    if "close" not in panel.columns or panel.n_rows < 2:
        return None
    close = panel.get("close")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = close[1:] / close[:-1]
        log_ret = np.log(ratio)
    mask = np.isfinite(log_ret) & (np.abs(log_ret) > np.log1p(opts.jump_threshold))
    if not mask.any():
        return None
    idx = np.flatnonzero(mask)[: opts.max_samples]
    samples = []
    for i in idx:
        r = float(ratio[int(i)])
        nearest = min(SPLIT_RATIOS, key=lambda c: abs(np.log(r / c)))
        near = abs(np.log(r / nearest)) <= opts.split_tolerance
        samples.append({"date": str(panel.dates[int(i) + 1]),
                        "ratio": round(r, 4),
                        "nearest_split": nearest if near else None})
    hints = sum(1 for s in samples if s["nearest_split"])
    return Finding("price_jump", "warn", int(mask.sum()),
                   f"有 {int(mask.sum())} 个单日涨跌超过 {opts.jump_threshold:.0%}"
                   + (f"（{hints} 个比例接近整数拆分）" if hints else ""),
                   "先确认是否未复权；拆股需要复权而不是删行",
                   samples)


def check_volume(panel: Panel, opts: Options) -> Optional[Finding]:
    if "volume" not in panel.columns:
        return None
    volume = panel.get("volume")
    mask = np.isfinite(volume) & (volume <= 0)
    if not mask.any():
        return None
    idx = np.flatnonzero(mask)[: opts.max_samples]
    return Finding("zero_volume", "info", int(mask.sum()),
                   f"有 {int(mask.sum())} 行成交量为 0 或负数",
                   "停牌日常见；若长期为 0 说明数据源有问题",
                   [_sample(panel, int(i)) for i in idx])


CHECKS = (check_nan, check_duplicates, check_non_positive, check_ohlc,
          check_gap, check_stale, check_outliers, check_jumps, check_volume)


def audit_panel(panel: Panel, opts: Optional[Options] = None) -> List[Finding]:
    """跑完所有体检项。"""
    opts = opts or Options()
    findings = []
    for check in CHECKS:
        result = check(panel, opts)
        if result is not None:
            findings.append(result)
    return findings


__all__ = ["CHECKS", "Finding", "Options", "SPLIT_RATIOS", "audit_panel"]