"""Kairos Clean —— 量化行情自动体检与清洗。

一行体检::

    import kairos_clean as kc
    report = kc.audit("data.csv")
    print(report["summary"]["score"])

一行清洗::

    result = kc.clean("data.csv", out_dir="cleaned", fill_gaps=True)
    print(result.summary())

只依赖 numpy，不上网。
"""
from __future__ import annotations

__version__ = "0.1.1"

from .api import CleanResult, audit, clean, quality_score  # noqa: E402
from .checks import Options  # noqa: E402
from .table import Panel, read_csv, write_csv  # noqa: E402

__all__ = ["CleanResult", "Options", "Panel", "__version__", "audit", "clean",
           "quality_score", "read_csv", "write_csv"]
