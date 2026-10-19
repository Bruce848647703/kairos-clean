"""体检与清洗的测试：每个问题都单独造一份脏数据。"""
from __future__ import annotations

import json

import numpy as np
import pytest

from kairos_clean import Panel
from kairos_clean.api import audit, clean, quality_score
from kairos_clean.checks import Options, audit_panel
from kairos_clean.report import changes_to_markdown, to_json, to_markdown
from kairos_clean.table import read_csv, write_csv


def business_days(n: int, start: str = "2024-01-01") -> np.ndarray:
    days = np.arange(np.datetime64(start), np.datetime64(start) + np.timedelta64(n * 2 + 10, "D"))
    return days[np.is_busday(days)][:n]


def make_panel(closes, *, symbol="TEST", volume=1000.0, ohlc=True) -> Panel:
    close = np.asarray(closes, dtype="float64")
    dates = business_days(close.size)
    columns = {"close": close}
    if ohlc:
        columns["open"] = close
        columns["high"] = close * 1.01
        columns["low"] = close * 0.99
        columns["volume"] = np.full(close.size, volume)
    return Panel(symbol, dates, columns)


def kinds(findings) -> set:
    return {f.kind for f in findings}


# --------------------------------------------------------------------------- #
# 读表
# --------------------------------------------------------------------------- #
def test_read_csv_basic_and_skips_bad_rows(tmp_path):
    path = tmp_path / "x.csv"
    path.write_text("date,open,high,low,close,volume\n"
                    "2024-01-02,10,11,9,10,100\n"
                    "bad-date,1,2,0.5,1,10\n"
                    "2024-01-03,10,12,9,11,200\n", encoding="utf-8")
    panels = read_csv(str(path))
    assert list(panels) == ["x"]
    panel = panels["x"]
    assert panel.n_rows == 2
    assert panel.skipped_rows == 1
    assert panel.get("close").tolist() == [10.0, 11.0]


def test_read_csv_column_aliases(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("日期,收盘价,成交量\n2024-01-02,10,100\n2024-01-03,11,200\n",
                    encoding="utf-8")
    panels = read_csv(str(path))
    panel = panels["a"]
    assert panel.get("close").tolist() == [10.0, 11.0]
    assert panel.get("volume").tolist() == [100.0, 200.0]


def test_read_csv_long_format_groups_symbols(tmp_path):
    path = tmp_path / "long.csv"
    path.write_text("date,symbol,close,volume\n"
                    "2024-01-02,AAA,10,1\n2024-01-02,BBB,20,2\n"
                    "2024-01-03,AAA,11,1\n2024-01-03,BBB,21,2\n", encoding="utf-8")
    panels = read_csv(str(path))
    assert set(panels) == {"AAA", "BBB"}
    assert panels["BBB"].get("close").tolist() == [20.0, 21.0]


def test_write_csv_roundtrip(tmp_path):
    panel = make_panel([10, 11, 12])
    path = tmp_path / "out.csv"
    write_csv(panel, str(path))
    again = read_csv(str(path))["out"]
    assert again.n_rows == panel.n_rows
    assert np.allclose(again.get("close"), panel.get("close"))


# --------------------------------------------------------------------------- #
# 单项检查
# --------------------------------------------------------------------------- #
def test_clean_panel_has_no_findings():
    panel = make_panel(np.linspace(10, 20, 100))
    assert audit_panel(panel) == []


def test_duplicate_dates():
    panel = make_panel([10, 11, 12, 13])
    panel.dates[2] = panel.dates[1]
    found = kinds(audit_panel(panel))
    assert "duplicate_dates" in found


def test_non_positive_price():
    closes = np.linspace(10, 20, 60)
    closes[30] = 0.0
    assert "non_positive_price" in kinds(audit_panel(make_panel(closes)))


def test_ohlc_violation():
    panel = make_panel(np.linspace(10, 20, 60))
    panel.columns["high"][10] = 1.0
    assert "ohlc_violation" in kinds(audit_panel(panel))


def test_calendar_gap():
    panel = make_panel(np.linspace(10, 20, 60))
    panel.dates[30:] += np.timedelta64(30, "D")
    finding = [f for f in audit_panel(panel) if f.kind == "calendar_gap"][0]
    assert finding.count >= 1
    assert finding.severity == "warn"


def test_stale_price():
    closes = np.linspace(10, 20, 60)
    closes[20:30] = closes[19]
    findings = audit_panel(make_panel(closes))
    assert "stale_price" in kinds(findings)


def test_return_outlier():
    rng = np.random.default_rng(0)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, 200)))
    closes[100:] *= 1.10                       # 10% 跳变，低于 25% 的 jump 阈值
    findings = audit_panel(make_panel(closes))
    assert "return_outlier" in kinds(findings)
    assert "price_jump" not in kinds(findings)


def test_price_jump_detects_split_ratio():
    closes = np.linspace(100, 120, 100)
    closes[50:] *= 0.5                          # 1 拆 2
    finding = [f for f in audit_panel(make_panel(closes)) if f.kind == "price_jump"][0]
    assert finding.count >= 1
    assert finding.samples[0]["nearest_split"] == 0.5


def test_zero_volume_and_nan():
    panel = make_panel(np.linspace(10, 20, 60))
    panel.columns["volume"][5] = 0.0
    panel.columns["close"][7] = np.nan
    found = kinds(audit_panel(panel))
    assert "zero_volume" in found
    assert "missing_values" in found


def test_thresholds_are_configurable():
    closes = np.linspace(10, 20, 60)
    closes[20:23] = closes[19]                 # 只连续 3 天
    assert "stale_price" not in kinds(audit_panel(make_panel(closes)))
    assert "stale_price" in kinds(
        audit_panel(make_panel(closes), Options(stale_days=3)))


def test_options_validation():
    with pytest.raises(ValueError):
        Options(stale_days=1)
    with pytest.raises(ValueError):
        Options(gap_days=0)
    with pytest.raises(ValueError):
        Options(outlier_k=0)


# --------------------------------------------------------------------------- #
# 质量分
# --------------------------------------------------------------------------- #
def test_quality_score_range_and_ordering():
    clean_panel = make_panel(np.linspace(10, 20, 100))
    dirty = make_panel(np.linspace(10, 20, 100))
    dirty.columns["close"][50] = -1.0
    dirty.columns["high"][30] = 0.1

    clean_score = quality_score(audit_panel(clean_panel), 100)
    dirty_score = quality_score(audit_panel(dirty), 100)
    assert clean_score == 100.0
    assert 0 <= dirty_score < clean_score


# --------------------------------------------------------------------------- #
# 清洗
# --------------------------------------------------------------------------- #
def test_clean_repairs_and_dedupes():
    panel = make_panel([10, 11, 12, 13, 14], volume=100)
    panel.columns["high"][1] = 0.5
    panel.dates[3] = panel.dates[2]

    result = clean(panel)
    cleaned = result.panels["TEST"]
    assert cleaned.n_rows == 4                       # 去重删了一行
    assert np.all(cleaned.get("high") >= np.maximum(cleaned.get("open"), cleaned.get("close")))
    assert any(c["kind"] == "ohlc_violation" for c in result.changes)
    assert any(c["kind"] == "duplicate_dates" for c in result.changes)


def test_clean_drops_bad_prices():
    closes = np.linspace(10, 20, 50)
    closes[10] = 0.0
    result = clean(make_panel(closes))
    assert result.panels["TEST"].n_rows == 49
    assert any(c["action"] == "drop" for c in result.changes)


def test_fill_gaps_adds_business_days():
    panel = make_panel(np.linspace(10, 20, 40))
    panel.dates[20:] += np.timedelta64(10, "D")
    result = clean(panel, fill_gaps=True)
    filled = result.panels["TEST"]
    assert filled.n_rows > 40
    assert np.all(filled.dates == np.sort(filled.dates))
    assert np.all(np.diff(filled.dates).astype(int) <= 3)   # 工作日序列：1 或 3 天
    assert any(c["kind"] == "calendar_gap" and c["action"] == "fill" for c in result.changes)


def test_winsorize_shrinks_outlier():
    rng = np.random.default_rng(1)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, 300)))
    closes[150:] *= 1.15
    result = clean(make_panel(closes), winsorize=True)
    after = result.after["symbols"]["TEST"]
    assert result.before["summary"]["issues"] > result.after["summary"]["issues"]
    assert not any(f["kind"] == "return_outlier" for f in after["findings"])
    assert any(c["action"] == "winsorize" for c in result.changes)


def test_clean_writes_files(tmp_path):
    panel = make_panel([10, 11, 12])
    out = tmp_path / "out"
    result = clean(panel, out_dir=str(out))
    assert (out / "TEST.csv").exists()
    assert result.paths["TEST"].endswith("TEST.csv")


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #
def test_report_markdown_and_json():
    panel = make_panel(np.linspace(10, 20, 100))
    panel.columns["high"][10] = 1.0
    report = audit(panel)
    md = to_markdown(report)
    assert "数据体检报告" in md and "质量分" in md
    assert "这些查不出来" in md
    parsed = json.loads(to_json(report))
    assert parsed["summary"]["rows"] == 100

    result = clean(panel)
    changes_md = changes_to_markdown(result.changes)
    assert "改动日志" in changes_md


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_cli_audit_and_clean(tmp_path, capsys):
    path = tmp_path / "dirty.csv"
    path.write_text("date,open,high,low,close,volume\n"
                    "2024-01-02,10,11,9,10,100\n"
                    "2024-01-03,10,10.5,9,10,100\n"     # high < open
                    "2024-01-04,0,0,0,0,0\n", encoding="utf-8")

    from kairos_clean.cli import main

    assert main(["audit", str(path)]) == 0
    out = capsys.readouterr().out
    assert "质量分" in out

    out_dir = tmp_path / "cleaned"
    assert main(["clean", str(path), "--out-dir", str(out_dir)]) == 0
    assert (out_dir / "dirty.csv").exists()
    assert (out_dir / "report.md").exists()


def test_cli_json_output(tmp_path, capsys):
    path = tmp_path / "ok.csv"
    path.write_text("date,close,volume\n2024-01-02,10,100\n2024-01-03,11,100\n",
                    encoding="utf-8")
    from kairos_clean.cli import main

    assert main(["audit", str(path), "--json"]) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["summary"]["rows"] == 2