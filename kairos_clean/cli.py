"""命令行：``kairos-clean audit`` / ``clean``。"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from . import __version__
from .api import audit, clean
from .checks import Options
from .report import changes_to_markdown, to_json, to_markdown


def _add_thresholds(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stale-days", type=int, default=5, help="连续多少天收盘价不变算停牌")
    parser.add_argument("--gap-days", type=int, default=7, help="相邻日期间隔超过多少天算缺口")
    parser.add_argument("--outlier-k", type=float, default=8.0, help="收益率稳健 z 阈值")
    parser.add_argument("--jump-threshold", type=float, default=0.25, help="单日涨跌多少算跳变")


def _options(args: argparse.Namespace) -> Options:
    return Options(stale_days=args.stale_days, gap_days=args.gap_days,
                   outlier_k=args.outlier_k, jump_threshold=args.jump_threshold)


def _print_summary(report: dict) -> None:
    summary = report["summary"]
    print(f"标的 {summary['n_symbols']} 个 · {summary['rows']:,} 行 · "
          f"问题 {summary['issues']:,} 个 · 质量分 {summary['score']:.1f}/100")
    sev = summary["by_severity"]
    print(f"错误 {sev['error']} · 警告 {sev['warn']} · 提示 {sev['info']}")


def _cmd_audit(args: argparse.Namespace) -> int:
    report = audit(args.file, options=_options(args))
    if args.json:
        print(to_json(report))
    else:
        _print_summary(report)
        print()
        for symbol, info in report["symbols"].items():
            if not info["findings"]:
                continue
            print(f"[{symbol or '单标的'}] 质量分 {info['score']:.1f}")
            for finding in info["findings"]:
                icon = {"error": "❌", "warn": "⚠️", "info": "ℹ️"}.get(finding["severity"], "")
                print(f"  {icon} {finding['kind']:<18} {finding['count']:>6}  "
                      f"{finding['message']}")
    if args.out:
        text = to_markdown(report)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"\n报告已写入 {args.out}")
    return 0


def _cmd_clean(args: argparse.Namespace) -> int:
    result = clean(
        args.file, out_dir=args.out_dir, options=_options(args),
        drop_duplicates=not args.keep_duplicates,
        repair_ohlc=not args.no_repair,
        fill_gaps=args.fill_gaps,
        winsorize=args.winsorize,
    )
    summary = result.summary()
    if args.json:
        print(to_json({"summary": summary, "changes": result.changes}))
    else:
        print(f"清洗完成：{summary['rows_before']:,} → {summary['rows_after']:,} 行，"
              f"改动 {summary['changes']} 处")
        print(f"质量分：{summary['score_before']:.1f} → {summary['score_after']:.1f}")
        for symbol, path in result.paths.items():
            print(f"  {symbol or 'data'} -> {path}")
        if result.changes:
            print("\n前 10 条改动：")
            for item in result.changes[:10]:
                print(f"  [{item['symbol'] or '-'}] {item['kind']}: {item['action']} {item['detail']}")
    if args.out_dir:
        report_path = os.path.join(args.out_dir, "report.md")
        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write(to_markdown(result.after, title="清洗后体检报告"))
            fh.write("\n")
            fh.write(changes_to_markdown(result.changes))
        print(f"\n报告已写入 {report_path}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kairos-clean",
        description="量化行情自动体检与清洗（日线 OHLCV）。")
    parser.add_argument("--version", action="version", version=f"kairos-clean {__version__}")
    sub = parser.add_subparsers(dest="command")

    audit_parser = sub.add_parser("audit", help="只体检，不改数据")
    audit_parser.add_argument("file")
    audit_parser.add_argument("--json", action="store_true")
    audit_parser.add_argument("--out", default="", help="可选：写一份 Markdown 报告")
    _add_thresholds(audit_parser)

    clean_parser = sub.add_parser("clean", help="清洗并输出干净 CSV")
    clean_parser.add_argument("file")
    clean_parser.add_argument("--out-dir", default="cleaned")
    clean_parser.add_argument("--fill-gaps", action="store_true", help="补工作日缺失行")
    clean_parser.add_argument("--winsorize", action="store_true", help="收益率 MAD 截尾")
    clean_parser.add_argument("--no-repair", action="store_true", help="不修 OHLC")
    clean_parser.add_argument("--keep-duplicates", action="store_true", help="不去重")
    clean_parser.add_argument("--json", action="store_true")
    _add_thresholds(clean_parser)

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    return {"audit": _cmd_audit, "clean": _cmd_clean}[args.command](args)


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())