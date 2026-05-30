# 💳 Loan Default Risk Analysis & Prediction

> An end-to-end machine learning pipeline to predict loan default probability with explainable AI — deployed as a real-time REST API with Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-AUC--ROC%3A0.88-green)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-orange)
![Docker](https://img.shields.io/badge/Docker-containerized-2496ED)

---

## 📌 Overview

Financial institutions lose billions annually to loan defaults. This project builds a production-ready ML classifier to predict default risk from applicant data — enabling faster, more accurate, and **explainable** credit decisions.

Key focus: not just accuracy, but **why** the model makes each prediction (SHAP explainability), making it business-actionable.

---

## 🏗️ ML Pipeline

```
Raw Data (150K+ records)
        │
        ▼
┌───────────────────┐
│  EDA & Analysis   │  ← Distribution plots, correlation heatmaps,
│                   │    default rate by feature
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   Preprocessing   │  ← Missing value imputation, outlier handling,
│                   │    label encoding, standard scaling
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Feature Engineering│  ← loan_to_income_ratio, employment_stability_score,
│                   │    debt_burden, monthly_payment
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  SMOTE Balancing  │  ← Handles 12.3% class imbalance
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   Model Training  │  ← LR, Random Forest, XGBoost, LightGBM
│   & Evaluation    │    Stratified K-Fold cross validation
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ SHAP Explainability│  ← Beeswarm plots, feature importance,
│                   │    per-prediction explanations
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  FastAPI + Docker │  ← Real-time prediction endpoint
│  Streamlit UI     │    with SHAP contributions per request
└───────────────────┘
```

---

## 📁 Project Structure

```
loan-default-risk-prediction/
│
├── requirements.txt
├── Dockerfile
│
├── src/
│   ├── data/
│   │   └── preprocessing.py        # Feature engineering, imputation, scaling, SMOTE
│   │
│   ├── models/
│   │   └── train.py                # Trains & benchmarks all 4 models, saves best
│   │
│   └── explainability/
│       └── shap_analysis.py        # SHAP beeswarm, bar plots, per-prediction explanations
│
├── api/
│   └── main.py                     # FastAPI: POST /predict with SHAP output
│
├── streamlit_app/
│   └── app.py                      # Interactive risk assessment dashboard
│
└── tests/
    └── test_preprocessing.py       # Unit tests for feature engineering
```

---

## 📊 Model Results

| Model | AUC-ROC | AUC-PR | F1 (Default) |
|-------|---------|--------|-------------|
| Logistic Regression | 0.74 | 0.41 | 0.52 |
| Random Forest | 0.83 | 0.59 | 0.67 |
| **XGBoost ✅** | **0.88** | **0.71** | **0.73** |
| LightGBM | 0.87 | 0.70 | 0.72 |

**Best model: XGBoost** with business-optimized threshold (0.35)

> Threshold tuned to 0.35 (vs default 0.5) to prioritize recall on defaults — reducing financial risk exposure.

---

## 🔍 Top Default Risk Factors (SHAP)

| Rank | Feature | Impact |
|------|---------|--------|
| 1 | Loan-to-income ratio | Highest positive driver of default |
| 2 | Credit history length | Long history = lower risk |
| 3 | Number of delinquencies | Strong positive signal |
| 4 | Employment stability score | Stable employment = lower risk |
| 5 | Debt-to-income ratio | High DTI increases default probability |

---

## ✨ Key Features

- **4 models trained & benchmarked** — LR, Random Forest, XGBoost, LightGBM
- **SMOTE** — handles 12.3% class imbalance without data leakage
- **SHAP explainability** — every prediction comes with feature contributions
- **Business-optimized threshold** — tuned for recall over precision
- **FastAPI endpoint** — real-time predictions with risk label + decision
- **Streamlit dashboard** — interactive risk assessment UI
- **Fully Dockerized** — single command deployment

---

## 🔌 Sample API Usage

**Request:**
```bash
curl -X POST "http://localhost:8001/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "loan_amount": 15000,
    "annual_income": 45000,
    "credit_score": 620,
    "debt_to_income": 0.45,
    "num_delinquencies": 2,
    "credit_history_months": 48,
    "loan_purpose": "debt_consolidation",
    "employment_type": "full_time",
    "employment_length": "1-3 years",
    "home_ownership": "RENT",
    "state": "CA",
    "open_accounts": 5
  }'
```

**Response:**
```json
{
  "default_probability": 0.6821,
  "risk_label": "High",
  "decision": "Decline",
  "top_risk_factors": [
    "loan_to_income_ratio",
    "num_delinquencies",
    "debt_to_income",
    "credit_score",
    "credit_history_months"
  ],
  "shap_contributions": {
    "loan_to_income_ratio": 0.2341,
    "num_delinquencies": 0.1823,
    "debt_to_income": 0.1456,
    "credit_score": -0.0921,
    "credit_history_months": -0.0654
  }
}
```

---

## 🚀 Setup & Installation

### Local Development

```bash
git clone https://github.com/ThokalaYashwanth/loan-default-risk-prediction
cd loan-default-risk-prediction
pip install -r requirements.txt

# Train the model first
python -m src.models.train

# Start the API
uvicorn api.main:app --reload --port 8001

# Start the Streamlit dashboard (separate terminal)
streamlit run streamlit_app/app.py
```

### Docker

```bash
docker build -t loan-default-api .
docker run -p 8001:8001 loan-default-api
```

- API: `http://localhost:8001`
- API Docs: `http://localhost:8001/docs`

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📈 Business Impact

- Reduces manual loan review time by flagging high-risk applications automatically
- SHAP explanations make decisions **auditable** — essential for regulatory compliance
- Business-threshold tuning minimizes false negatives (missed defaults) which carry higher cost than false positives

---

## 🌱 Future Improvements

- [ ] Add LightGBM as production model (marginal AUC difference, 3x faster inference)
- [ ] Implement model drift monitoring
- [ ] Add CI/CD with GitHub Actions + automated retraining
- [ ] Build credit score simulation tool for applicants

---

## 👤 Author

**Thokala Yashwanth**
- 📧 thokalayashwanth143@gmail.com
- 🔗 [LinkedIn](https://linkedin.com/in/thokalayashwanth)
- 💻 [GitHub](https://github.com/ThokalaYashwanth)
