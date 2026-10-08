"""Smoke tests: the Streamlit dashboard and its help text render without errors."""

from pathlib import Path

import pandas as pd
import pytest

from glossary import COLUMN_HELP, MODEL_INFO, TERMS
from goldpred.models import MODEL_NAMES
from goldpred.report import format_table

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_renders_with_defaults():
    at = AppTest.from_file(APP, default_timeout=300).run()
    assert not at.exception
    assert len(at.metric) >= 5
    assert all(m.help for m in at.metric), "every metric should have a tooltip"
    assert "Gold Price Direction Forecasting" in at.title[0].value
    assert len(at.dataframe) == 1  # model comparison table
    assert len(at.tabs) == 5


def test_app_handles_other_settings():
    at = AppTest.from_file(APP, default_timeout=300).run()
    at.selectbox[0].set_value("Logistic Regression")
    at.slider[0].set_value(2016)  # first out-of-sample year
    at.slider[1].set_value(0.55)  # stricter entry threshold
    at.slider[2].set_value(0.0)  # zero transaction costs
    at.run()
    assert not at.exception


def test_glossary_search():
    at = AppTest.from_file(APP, default_timeout=300).run()
    at.text_input[0].set_value("sharpe").run()
    assert not at.exception
    shown = " ".join(m.value for m in at.markdown)
    assert "**Sharpe ratio**" in shown
    assert "**Walk-forward validation**" not in shown


def test_help_text_covers_every_model_and_column():
    assert set(MODEL_INFO) == set(MODEL_NAMES)
    columns = format_table(pd.DataFrame(columns=[
        "accuracy", "accuracy_ci95", "roc_auc", "cagr", "sharpe", "max_drawdown", "exposure", "trades",
    ])).columns
    assert set(COLUMN_HELP) == set(columns)
    assert all(text.strip() for text in TERMS.values())
