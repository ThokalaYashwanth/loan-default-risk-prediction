from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import pandas as pd
import numpy as np
import os
from src.explainability.shap_analysis import explain_single

app = FastAPI(
    title="Loan Default Risk Prediction API",
    description="Predicts loan default probability using XGBoost with SHAP explainability",
    version="1.0.0",
)

ARTIFACTS_DIR = "src/models/artifacts"
THRESHOLD = 0.35

# Load model at startup
model = None
try:
    model = joblib.load(f"{ARTIFACTS_DIR}/best_model.pkl")
    print("[API] Model loaded successfully")
except FileNotFoundError:
    print("[API] Warning: model not found. Run training first.")


class LoanApplication(BaseModel):
    loan_amount: float = Field(..., example=15000, description="Requested loan amount in USD")
    annual_income: float = Field(..., example=55000)
    debt_to_income: float = Field(..., example=0.32, description="DTI ratio 0-1")
    credit_score: int = Field(..., example=680, ge=300, le=850)
    num_delinquencies: int = Field(0, example=1)
    credit_history_months: int = Field(..., example=84)
    loan_purpose: str = Field(..., example="debt_consolidation")
    employment_type: str = Field(..., example="full_time")
    home_ownership: str = Field(..., example="RENT")
    state: str = Field(..., example="CA")
    open_accounts: int = Field(..., example=7)
    employment_length: Optional[str] = Field("3-5 years", example="3-5 years")


class PredictionResponse(BaseModel):
    default_probability: float
    risk_label: str           # Low / Medium / High
    decision: str             # Approve / Review / Decline
    top_risk_factors: list
    shap_contributions: dict


@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    raw = application.dict()
    df = pd.DataFrame([raw])

    # Feature engineering (same as training)
    df["loan_to_income_ratio"] = df["loan_amount"] / (df["annual_income"] + 1)
    df["monthly_payment"] = df["loan_amount"] / 36
    df["debt_burden"] = df["debt_to_income"] * df["annual_income"] / 12
    stability_map = {"<1 year": 1, "1-3 years": 2, "3-5 years": 3, "5-10 years": 4, ">10 years": 5}
    df["employment_stability_score"] = df["employment_length"].map(stability_map).fillna(2)

    from src.data.preprocessing import NUMERIC_COLS, CATEGORICAL_COLS
    import joblib

    num_imputer = joblib.load(f"{ARTIFACTS_DIR}/num_imputer.pkl")
    scaler = joblib.load(f"{ARTIFACTS_DIR}/scaler.pkl")
    df[NUMERIC_COLS] = num_imputer.transform(df[NUMERIC_COLS])

    for col in CATEGORICAL_COLS:
        enc = joblib.load(f"{ARTIFACTS_DIR}/encoder_{col}.pkl")
        df[col] = enc.transform(df[col].astype(str))

    df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])
    X = df[NUMERIC_COLS + CATEGORICAL_COLS]

    proba = float(model.predict_proba(X)[0][1])

    # Risk label
    if proba < 0.25:
        risk_label, decision = "Low", "Approve"
    elif proba < 0.50:
        risk_label, decision = "Medium", "Review"
    else:
        risk_label, decision = "High", "Decline"

    # SHAP explanation (lightweight — skip if slow)
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
        if isinstance(sv, list):
            sv = sv[1]
        contributions = pd.Series(sv[0], index=X.columns)
        top = contributions.abs().sort_values(ascending=False).head(5)
        shap_dict = contributions[top.index].round(4).to_dict()
        top_features = top.index.tolist()
    except Exception:
        shap_dict = {}
        top_features = []

    return PredictionResponse(
        default_probability=round(proba, 4),
        risk_label=risk_label,
        decision=decision,
        top_risk_factors=top_features,
        shap_contributions=shap_dict,
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8001, reload=True)
