import numpy as np
import os
import joblib
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

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

# === Feature names ===
channels = ["F4A1", "C4A1", "O2A1"]
base_feats = ["delta", "theta", "alpha", "beta", "gamma", "δ/θ", "δ/α", "α/β", "θ/α",
              "activity", "mobility", "complexity", "mean", "std", "skew"]
feature_names = [f"{ch}_{f}" for ch in channels for f in base_feats]
xgb_idx = [feature_names.index(f) for f in xgb_features]

# === CV setup ===
gkf = GroupKFold(n_splits=5)

rf_scores, xgb_scores, ensemble_scores = [], [], []

for train_idx, test_idx in gkf.split(X, y, groups):
    # Features per model
    X_train_rf, X_test_rf = X[train_idx][:, rf_idx], X[test_idx][:, rf_idx]
    X_train_xgb, X_test_xgb = X[train_idx][:, xgb_idx], X[test_idx][:, xgb_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # --- Train & predict RF
    rf_model.fit(X_train_rf, y_train)
    rf_probs = rf_model.predict_proba(X_test_rf)[:, 1]
    rf_preds = (rf_probs >= rf_threshold).astype(int)
    rf_scores.append(accuracy_score(y_test, rf_preds))

    # --- Train & predict XGB
    xgb_model.fit(X_train_xgb, y_train)
    xgb_probs = xgb_model.predict_proba(X_test_xgb)[:, 1]
    xgb_preds = (xgb_probs >= xgb_threshold).astype(int)
    xgb_scores.append(accuracy_score(y_test, xgb_preds))

    # --- Ensemble (average probs)
    avg_probs = (rf_probs + xgb_probs) / 2.0
    ensemble_preds = (avg_probs >= 0.5).astype(int)  # threshold can be tuned
    ensemble_scores.append(accuracy_score(y_test, ensemble_preds))

# === Results ===
rf_scores = np.array(rf_scores)
xgb_scores = np.array(xgb_scores)
ensemble_scores = np.array(ensemble_scores)

print("\n--- CV Results ---")
print("RF Accuracies:", rf_scores, "Mean =", rf_scores.mean())
print("XGB Accuracies:", xgb_scores, "Mean =", xgb_scores.mean())
print("Ensemble Accuracies:", ensemble_scores, "Mean =", ensemble_scores.mean())

if ensemble_scores.mean() > max(rf_scores.mean(), xgb_scores.mean()):
    print("\n✅ Ensemble beats both!")
else:
    print("\n⚖️ Ensemble does not improve over best individual model")