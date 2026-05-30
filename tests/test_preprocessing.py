import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.data.preprocessing import engineer_features, NUMERIC_COLS, CATEGORICAL_COLS


def make_sample_df(n=10) -> pd.DataFrame:
    np.random.seed(42)
    return pd.DataFrame({
        "loan_amount": np.random.randint(5000, 50000, n),
        "annual_income": np.random.randint(30000, 150000, n),
        "debt_to_income": np.random.uniform(0.1, 0.6, n),
        "credit_score": np.random.randint(500, 800, n),
        "num_delinquencies": np.random.randint(0, 5, n),
        "credit_history_months": np.random.randint(12, 240, n),
        "loan_purpose": ["debt_consolidation"] * n,
        "employment_type": ["full_time"] * n,
        "home_ownership": ["RENT"] * n,
        "state": ["CA"] * n,
        "open_accounts": np.random.randint(2, 15, n),
        "employment_length": ["3-5 years"] * n,
        "default": np.random.randint(0, 2, n),
    })


class TestFeatureEngineering:
    def test_loan_to_income_ratio(self):
        df = make_sample_df()
        out = engineer_features(df)
        assert "loan_to_income_ratio" in out.columns
        expected = df["loan_amount"] / (df["annual_income"] + 1)
        pd.testing.assert_series_equal(out["loan_to_income_ratio"], expected)

    def test_monthly_payment(self):
        df = make_sample_df()
        out = engineer_features(df)
        assert "monthly_payment" in out.columns
        np.testing.assert_allclose(out["monthly_payment"].values, df["loan_amount"].values / 36)

    def test_employment_stability_score(self):
        df = make_sample_df()
        out = engineer_features(df)
        assert "employment_stability_score" in out.columns
        assert out["employment_stability_score"].iloc[0] == 3  # "3-5 years" → 3

    def test_no_nan_in_derived(self):
        df = make_sample_df()
        out = engineer_features(df)
        for col in ["loan_to_income_ratio", "monthly_payment", "debt_burden"]:
            assert out[col].isna().sum() == 0, f"{col} has NaNs"


class TestColumnSchema:
    def test_numeric_cols_defined(self):
        assert len(NUMERIC_COLS) > 0
        assert "loan_amount" in NUMERIC_COLS
        assert "credit_score" in NUMERIC_COLS

    def test_categorical_cols_defined(self):
        assert len(CATEGORICAL_COLS) > 0
        assert "loan_purpose" in CATEGORICAL_COLS
        assert "state" in CATEGORICAL_COLS


class TestDataIntegrity:
    def test_sample_df_shape(self):
        df = make_sample_df(50)
        assert df.shape == (50, 13)

    def test_default_column_binary(self):
        df = make_sample_df(100)
        assert set(df["default"].unique()).issubset({0, 1})

    def test_credit_score_range(self):
        df = make_sample_df(200)
        assert df["credit_score"].between(300, 850).all()
