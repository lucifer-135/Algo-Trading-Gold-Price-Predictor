"""Smoke test: the Streamlit dashboard renders every tab without errors."""

from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_renders_with_defaults():
    at = AppTest.from_file(APP, default_timeout=300).run()
    assert not at.exception
    assert len(at.metric) >= 5
    assert "Gold Price Direction Forecasting" in at.title[0].value
    assert len(at.dataframe) == 1  # model comparison table


def test_app_handles_other_settings():
    at = AppTest.from_file(APP, default_timeout=300).run()
    at.selectbox[0].set_value("Logistic Regression")
    at.slider[0].set_value(2016)  # first out-of-sample year
    at.slider[1].set_value(0.55)  # stricter entry threshold
    at.slider[2].set_value(0.0)  # zero transaction costs
    at.run()
    assert not at.exception
