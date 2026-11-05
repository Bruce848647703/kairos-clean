"""表格读入/写出：把 CSV 变成一个轻量的 Panel（numpy 数组 + 日期）。

不做复杂的事情，只负责：
- 认列名（date/日期, close/收盘 ... 大小写、空格都宽容）；
- 日期解析失败的行直接跳过并计数；
- 长表（带 symbol 列）自动按标的分组。
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np

DATE_ALIASES = ("date", "datetime", "time", "日期", "交易日期", "trade_date")
SYMBOL_ALIASES = ("symbol", "ticker", "code", "代码", "证券代码", "标的")
PRICE_ALIASES = {
    "open": ("open", "开盘", "开盘价"),
    "high": ("high", "最高", "最高价"),
    "low": ("low", "最低", "最低价"),
    "close": ("close", "收盘", "收盘价", "price", "价格"),
    "volume": ("volume", "vol", "成交量", "成交额"),
}


@dataclass
class Panel:
    """一个标的的序列表格：日期 + 若干数值列。"""

    symbol: str
    dates: np.ndarray
    columns: Dict[str, np.ndarray] = field(default_factory=dict)
    skipped_rows: int = 0

    def __post_init__(self) -> None:
        self.dates = np.asarray(self.dates, dtype="datetime64[D]")
        for name, values in list(self.columns.items()):
            self.columns[name] = np.asarray(values, dtype="float64")
        if self.dates.ndim != 1:
            raise ValueError("dates 必须是一维")
        for name, values in self.columns.items():
            if values.shape != self.dates.shape:
                raise ValueError(f"列 {name} 长度与日期不一致")

    @property
    def n_rows(self) -> int:
        return int(self.dates.size)

    def has(self, *names: str) -> bool:
        return all(name in self.columns for name in names)

    def get(self, name: str) -> np.ndarray:
        if name not in self.columns:
            raise KeyError(f"{self.symbol} 没有列 {name!r}")
        return self.columns[name]

    def slice(self, indices) -> "Panel":
        idx = np.asarray(indices, dtype="int64")
        return Panel(self.symbol, self.dates[idx],
                     {k: v[idx] for k, v in self.columns.items()})


def _norm(text: str) -> str:
    return str(text).strip().lower().replace(" ", "")


def _pick(header: Sequence[str], aliases: Sequence[str]) -> Optional[int]:
    lowered = [_norm(h) for h in header]
    for alias in aliases:
        if _norm(alias) in lowered:
            return lowered.index(_norm(alias))
    return None


def _to_float(text: str) -> float:
    try:
        value = float(str(text).strip().replace(",", ""))
    except (TypeError, ValueError):
        return float("nan")
    return value


def read_csv(path: str, *, symbol: Optional[str] = None) -> Dict[str, Panel]:
    """读 CSV，返回 ``{symbol: Panel}``（单标的时 key 为文件名）。"""
    stem = os.path.splitext(os.path.basename(path))[0]
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"{path} 是空文件") from None
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} 只有表头")

    date_idx = _pick(header, DATE_ALIASES)
    if date_idx is None:
        raise ValueError(f"{path} 找不到日期列，实际列：{header}")
    sym_idx = _pick(header, SYMBOL_ALIASES)

    numeric: Dict[str, int] = {}
    for canonical, aliases in PRICE_ALIASES.items():
        idx = _pick(header, aliases)
        if idx is not None:
            numeric[canonical] = idx
    if "close" not in numeric and "volume" not in numeric:
        raise ValueError(f"{path} 找不到收盘价/成交量列，实际列：{header}")

    buckets: Dict[str, Dict[str, list]] = {}
    skipped = 0
    for row in rows:
        if len(row) <= date_idx:
            skipped += 1
            continue
        try:
            day = np.datetime64(row[date_idx].strip()[:10], "D")
        except (ValueError, TypeError):
            skipped += 1
            continue
        if np.isnat(day):
            skipped += 1
            continue
        key = symbol or (row[sym_idx].strip() if sym_idx is not None and sym_idx < len(row) else stem)
        slot = buckets.setdefault(key, {"dates": [], **{k: [] for k in numeric}})
        slot["dates"].append(day)
        for canonical, idx in numeric.items():
            value = _to_float(row[idx]) if idx < len(row) else float("nan")
            slot[canonical].append(value)

    panels: Dict[str, Panel] = {}
    for key, slot in buckets.items():
        dates = np.asarray(slot["dates"], dtype="datetime64[D]")
        columns = {k: np.asarray(slot[k], dtype="float64") for k in numeric}
        order = np.argsort(dates, kind="stable")
        panels[key] = Panel(key, dates[order],
                            {k: v[order] for k, v in columns.items()}, skipped)
    return panels


def write_csv(panel: Panel, path: str) -> str:
    """把 Panel 写回 CSV（列顺序：date, 其余按名字）。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    names = ["date"] + list(panel.columns)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(names)
        for i in range(panel.n_rows):
            row = [str(panel.dates[i])]
            for name in panel.columns:
                value = panel.columns[name][i]
                row.append("" if not np.isfinite(value) else f"{value:.6f}")
            writer.writerow(row)
    return path


__all__ = ["Panel", "read_csv", "write_csv"]