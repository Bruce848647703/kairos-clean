"""把体检/清洗结果渲染成 Markdown 或 JSON。"""
from __future__ import annotations

import json
from typing import Any, Dict, List

SEVERITY_ICON = {"error": "❌", "warn": "⚠️", "info": "ℹ️"}
SEVERITY_LABEL = {"error": "错误", "warn": "警告", "info": "提示"}


def _f(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def to_markdown(report: Dict[str, Any], *, title: str = "数据体检报告") -> str:
    summary = report.get("summary", {})
    lines = [
        f"# {title}", "",
        f"- 标的数：{summary.get('n_symbols', 0)}",
        f"- 总行数：{summary.get('rows', 0):,}",
        f"- 问题数：{summary.get('issues', 0):,}",
        f"- 质量分：**{_f(summary.get('score', 0.0), 1)} / 100**",
        "",
        "## 分标的概览", "",
        "| 标的 | 行数 | 区间 | 问题 | 质量分 |",
        "|---|---:|---|---:|---:|",
    ]
    for symbol, info in report.get("symbols", {}).items():
        span = f"{info['start']} → {info['end']}" if info.get("start") else "—"
        lines.append(f"| {symbol or '（单标的）'} | {info['rows']:,} | {span} | "
                     f"{info['issues']:,} | {_f(info['score'], 1)} |")

    lines += ["", "## 明细", ""]
    for symbol, info in report.get("symbols", {}).items():
        if not info["findings"]:
            continue
        lines += [f"### {symbol or '（单标的）'}", "",
                  "| 严重度 | 问题 | 数量 | 说明 | 建议 |",
                  "|---|---|---:|---|---|"]
        for finding in info["findings"]:
            icon = SEVERITY_ICON.get(finding["severity"], "")
            label = SEVERITY_LABEL.get(finding["severity"], finding["severity"])
            lines.append(
                f"| {icon} {label} | `{finding['kind']}` | {finding['count']} | "
                f"{finding['message']} | {finding['suggestion']} |")
        lines.append("")
        samples = [f for f in info["findings"] if f["samples"]]
        if samples:
            lines.append("<details><summary>样例</summary>")
            lines.append("")
            for finding in samples:
                lines.append(f"**{finding['kind']}**")
                for sample in finding["samples"]:
                    lines.append(f"- `{sample}`")
                lines.append("")
            lines.append("</details>")
            lines.append("")

    lines += [
        "## 说明（这些查不出来）", "",
        "- 只按**工作日**判断缺行，法定节假日会被当成缺口（除非你传入真实交易日历）；",
        "- 拆股只给「嫌疑」不自动复权 —— 复权涉及历史全序列，擅自改会毁掉研究；",
        "- 异常值用 MAD 稳健 z 分数，慢趋势里的正常大幅波动可能被标出来；",
        "- 时间戳只认日期（日线），分钟线请先切分到日。",
        "",
    ]
    return "\n".join(lines)


def to_json(report: Dict[str, Any], *, indent: int = 2) -> str:
    return json.dumps(report, ensure_ascii=False, indent=indent)


def changes_to_markdown(changes: List[Dict[str, Any]], limit: int = 200) -> str:
    if not changes:
        return "_没有做任何修改。_\n"
    lines = ["# 改动日志", "", "| 标的 | 类型 | 动作 | 说明 |", "|---|---|---|---|"]
    for item in changes[:limit]:
        lines.append(f"| {item['symbol'] or '—'} | `{item['kind']}` | "
                     f"{item['action']} | {item['detail']} |")
    if len(changes) > limit:
        lines.append(f"| … | | | 还有 {len(changes) - limit} 条 |")
    return "\n".join(lines) + "\n"


__all__ = ["changes_to_markdown", "to_json", "to_markdown"]