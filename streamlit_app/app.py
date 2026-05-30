import streamlit as st
import requests
import pandas as pd

API_URL = "http://localhost:8001/predict"

st.set_page_config(page_title="Loan Default Predictor", page_icon="💳", layout="wide")
st.title("💳 Loan Default Risk Predictor")
st.caption("XGBoost · SHAP Explainability · FastAPI")

st.sidebar.header("Loan Application Details")

loan_amount = st.sidebar.number_input("Loan Amount ($)", 1000, 100000, 15000, 500)
annual_income = st.sidebar.number_input("Annual Income ($)", 10000, 500000, 55000, 1000)
credit_score = st.sidebar.slider("Credit Score", 300, 850, 680)
debt_to_income = st.sidebar.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.32, 0.01)
num_delinquencies = st.sidebar.number_input("# Delinquencies (last 2yr)", 0, 20, 1)
credit_history_months = st.sidebar.number_input("Credit History (months)", 6, 360, 84)
open_accounts = st.sidebar.number_input("Open Accounts", 1, 30, 7)
loan_purpose = st.sidebar.selectbox("Loan Purpose", ["debt_consolidation", "home_improvement", "medical", "car", "vacation", "other"])
employment_type = st.sidebar.selectbox("Employment Type", ["full_time", "part_time", "self_employed", "unemployed"])
employment_length = st.sidebar.selectbox("Employment Length", ["<1 year", "1-3 years", "3-5 years", "5-10 years", ">10 years"])
home_ownership = st.sidebar.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE"])
state = st.sidebar.text_input("State (2-letter)", "CA").upper()

if st.sidebar.button("🔍 Predict Default Risk", use_container_width=True):
    payload = {
        "loan_amount": loan_amount,
        "annual_income": annual_income,
        "credit_score": credit_score,
        "debt_to_income": debt_to_income,
        "num_delinquencies": num_delinquencies,
        "credit_history_months": credit_history_months,
        "open_accounts": open_accounts,
        "loan_purpose": loan_purpose,
        "employment_type": employment_type,
        "employment_length": employment_length,
        "home_ownership": home_ownership,
        "state": state,
    }

    with st.spinner("Running prediction..."):
        try:
            resp = requests.post(API_URL, json=payload, timeout=15)
            data = resp.json()

            proba = data["default_probability"]
            risk = data["risk_label"]
            decision = data["decision"]

            col1, col2, col3 = st.columns(3)
            col1.metric("Default Probability", f"{proba:.1%}")
            col2.metric("Risk Level", risk)
            color = {"Approve": "success", "Review": "warning", "Decline": "error"}[decision]
            getattr(col3, color)(f"Decision: **{decision}**")

            st.progress(proba)

            if data.get("top_risk_factors"):
                st.subheader("Top Risk Factors (SHAP)")
                shap_vals = data.get("shap_contributions", {})
                df_shap = pd.DataFrame({
                    "Feature": data["top_risk_factors"],
                    "SHAP Contribution": [shap_vals.get(f, 0) for f in data["top_risk_factors"]],
                })
                st.bar_chart(df_shap.set_index("Feature"))

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Start the FastAPI server first.")
