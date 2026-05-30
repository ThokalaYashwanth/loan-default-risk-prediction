import shap
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

ARTIFACTS_DIR = "src/models/artifacts"
PLOTS_DIR = "outputs/shap_plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


def get_explainer(model, X_background: pd.DataFrame):
    """Create a SHAP TreeExplainer for tree-based models."""
    return shap.TreeExplainer(model, X_background)


def compute_shap_values(explainer, X: pd.DataFrame):
    return explainer.shap_values(X)


def plot_summary(shap_values, X: pd.DataFrame, save_path: str = None):
    """Beeswarm summary plot — feature importance across all samples."""
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    path = save_path or f"{PLOTS_DIR}/summary_beeswarm.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Saved beeswarm plot → {path}")
    return path


def plot_bar_importance(shap_values, X: pd.DataFrame, save_path: str = None):
    """Global feature importance bar chart."""
    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.tight_layout()
    path = save_path or f"{PLOTS_DIR}/feature_importance_bar.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Saved bar plot → {path}")
    return path


def explain_single(model, X_background: pd.DataFrame, X_instance: pd.DataFrame) -> dict:
    """Return SHAP feature contributions for a single prediction."""
    explainer = get_explainer(model, X_background)
    sv = explainer.shap_values(X_instance)

    if isinstance(sv, list):
        sv = sv[1]  # for binary classifiers that return list

    contributions = pd.Series(sv[0], index=X_instance.columns)
    top = contributions.abs().sort_values(ascending=False).head(5)

    return {
        "top_features": top.index.tolist(),
        "shap_values": contributions[top.index].to_dict(),
        "base_value": float(explainer.expected_value if not isinstance(explainer.expected_value, list)
                            else explainer.expected_value[1]),
    }


def run_full_explainability(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Run full SHAP analysis on the best model."""
    model = joblib.load(f"{ARTIFACTS_DIR}/best_model.pkl")
    explainer = get_explainer(model, X_train.sample(min(200, len(X_train)), random_state=42))
    shap_values = compute_shap_values(explainer, X_test)

    plot_summary(shap_values, X_test)
    plot_bar_importance(shap_values, X_test)

    # Mean absolute SHAP per feature
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False)

    print("\n[SHAP] Top default risk factors:")
    print(importance_df.head(10).to_string(index=False))
    return importance_df
