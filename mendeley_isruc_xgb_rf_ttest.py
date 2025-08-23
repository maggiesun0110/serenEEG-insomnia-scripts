import numpy as np
import os
import joblib
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import ttest_rel

# === Paths ===
results_path = os.path.join("..", "results")
data = np.load(os.path.join(results_path, "combined_18fts.npz"))
X, y, groups = data["features"], data["labels"], data["subject_ids"]

# === Load models ===
xgb_bundle = joblib.load(os.path.join(results_path, "xgb_rfe_tuned_model_better.joblib"))
rf_bundle = joblib.load(os.path.join(results_path, "rf_rfe_tuned_model_better.joblib"))

xgb_model = xgb_bundle["model"]
xgb_threshold = xgb_bundle["threshold"]
xgb_features = xgb_bundle["selected_features"]

rf_model = rf_bundle["model"]
rf_threshold = rf_bundle["threshold"]
rf_idx = rf_bundle["selected_idx"]

# === Feature selection ===
channels = ["F4A1", "C4A1", "O2A1"]
base_feats = ["delta", "theta", "alpha", "beta", "gamma", "δ/θ", "δ/α", "α/β", "θ/α",
              "activity", "mobility", "complexity", "mean", "std", "skew"]
feature_names = [f"{ch}_{f}" for ch in channels for f in base_feats]

# Map back to indices for XGB’s feature list
xgb_idx = [feature_names.index(f) for f in xgb_features]

# === CV setup ===
gkf = GroupKFold(n_splits=5)

rf_scores, xgb_scores = [], []

for train_idx, test_idx in gkf.split(X, y, groups):
    X_train_rf, X_test_rf = X[train_idx][:, rf_idx], X[test_idx][:, rf_idx]
    X_train_xgb, X_test_xgb = X[train_idx][:, xgb_idx], X[test_idx][:, xgb_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # RF predictions
    rf_model.fit(X_train_rf, y_train)
    rf_probs = rf_model.predict_proba(X_test_rf)[:, 1]
    rf_preds = (rf_probs >= rf_threshold).astype(int)
    rf_scores.append(accuracy_score(y_test, rf_preds))

    # XGB predictions
    xgb_model.fit(X_train_xgb, y_train)
    xgb_probs = xgb_model.predict_proba(X_test_xgb)[:, 1]
    xgb_preds = (xgb_probs >= xgb_threshold).astype(int)
    xgb_scores.append(accuracy_score(y_test, xgb_preds))

# === Results ===
rf_scores = np.array(rf_scores)
xgb_scores = np.array(xgb_scores)

print("RF CV Accuracies:", rf_scores)
print("XGB CV Accuracies:", xgb_scores)
print(f"RF mean = {rf_scores.mean():.4f}")
print(f"XGB mean = {xgb_scores.mean():.4f}")

# === Paired t-test ===
t_stat, p_val = ttest_rel(xgb_scores, rf_scores)
print("\nPaired t-test results:")
print(f"T-statistic = {t_stat:.4f}, P-value = {p_val:.4f}")

if p_val < 0.05:
    if xgb_scores.mean() > rf_scores.mean():
        print("✅ XGB significantly outperforms RF")
    else:
        print("✅ RF significantly outperforms XGB")
else:
    print("⚖️ No statistically significant difference")