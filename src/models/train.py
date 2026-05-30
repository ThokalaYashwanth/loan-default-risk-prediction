import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, classification_report, confusion_matrix,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from src.data.preprocessing import preprocess, apply_smote, load_data

ARTIFACTS_DIR = "src/models/artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
    "xgboost": XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="auc", random_state=42,
    ),
    "lightgbm": LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, random_state=42, verbose=-1,
    ),
}


def evaluate(model, X_test, y_test, name: str, threshold: float = 0.35) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    auc = roc_auc_score(y_test, proba)
    auc_pr = average_precision_score(y_test, proba)
    f1 = f1_score(y_test, preds)

    print(f"\n{'='*50}")
    print(f"Model: {name}")
    print(f"  AUC-ROC : {auc:.4f}")
    print(f"  AUC-PR  : {auc_pr:.4f}")
    print(f"  F1 (def): {f1:.4f} (threshold={threshold})")
    print(classification_report(y_test, preds, target_names=["Non-default", "Default"]))

    return {"name": name, "auc_roc": auc, "auc_pr": auc_pr, "f1_default": f1}


def train(data_path: str = "data/loan_data.csv"):
    df = load_data(data_path)
    X, y = preprocess(df, fit=True, artifacts_dir=ARTIFACTS_DIR)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Apply SMOTE only on training data
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    results = []
    for name, model in MODELS.items():
        print(f"\n[Train] Fitting {name}...")
        model.fit(X_train_res, y_train_res)
        result = evaluate(model, X_test, y_test, name)
        results.append(result)

        joblib.dump(model, f"{ARTIFACTS_DIR}/{name}.pkl")

    # Save best model (by AUC-ROC)
    best = max(results, key=lambda r: r["auc_roc"])
    print(f"\n✅ Best model: {best['name']} — AUC-ROC: {best['auc_roc']:.4f}")
    best_model = joblib.load(f"{ARTIFACTS_DIR}/{best['name']}.pkl")
    joblib.dump(best_model, f"{ARTIFACTS_DIR}/best_model.pkl")

    return results


if __name__ == "__main__":
    train()
