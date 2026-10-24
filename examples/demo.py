"""造一份脏数据，然后体检 + 清洗，看看前后差多少。

运行：python examples/demo.py
"""
from __future__ import annotations

import os

import numpy as np

import kairos_clean as kc

OUT = os.path.join("examples", "out")


def make_dirty(n: int = 120, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    closes = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, n)))

    closes[20] = 0.0            # 坏价格
    closes[40:46] = closes[39]  # 停牌（连续 6 天不变）
    closes[70:] *= 0.5          # 1 拆 2，没复权
    closes[95] *= 1.10          # 单日异常
    return closes


def write_csv(path: str, closes: np.ndarray) -> None:
    days = np.arange(np.datetime64("2024-01-01"),
                     np.datetime64("2024-01-01") + np.timedelta64(len(closes) * 2, "D"))
    days = days[np.is_busday(days)][: len(closes)]
    days = np.delete(days, 60)                  # 人为漏一天
    closes = np.delete(closes, 60)
    days = np.append(days, days[80])            # 重复日期
    closes = np.append(closes, closes[80])

    os.makedirs(OUT, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("date,open,high,low,close,volume\n")
        for i, (day, close) in enumerate(zip(days, closes)):
            high = close * 1.01 if i != 30 else close * 0.5   # OHLC 矛盾
            low = close * 0.99
            volume = 0.0 if i == 90 else 1_000_000
            fh.write(f"{day},{close:.4f},{high:.4f},{low:.4f},"
                     f"{close:.4f},{volume:.0f}\n")


def main() -> None:
    path = os.path.join(OUT, "dirty.csv")
    write_csv(path, make_dirty())

    report = kc.audit(path)
    print("=== 清洗前 ===")
    print(f"行数 {report['summary']['rows']} · 问题 {report['summary']['issues']} "
          f"· 质量分 {report['summary']['score']:.1f}")
    for finding in report["symbols"]["dirty"]["findings"]:
        print(f"  {finding['severity']:<5} {finding['kind']:<20} {finding['count']:>3}  "
              f"{finding['message']}")

    result = kc.clean(path, out_dir=os.path.join(OUT, "cleaned"),
                      fill_gaps=True, winsorize=True)
    summary = result.summary()
    print("\n=== 清洗后 ===")
    print(f"行数 {summary['rows_before']} → {summary['rows_after']}，"
          f"改动 {summary['changes']} 处")
    print(f"质量分 {summary['score_before']:.1f} → {summary['score_after']:.1f}")
    print("\n前 8 条改动：")
    for item in result.changes[:8]:
        print(f"  [{item['kind']}] {item['action']} {item['detail']}")

    print(f"\n干净数据: {result.paths['dirty']}")
    print(f"报告: {os.path.join(OUT, 'cleaned', 'report.md')}")


if __name__ == "__main__":
    main()
