import numpy as np
import os
import joblib
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import GroupShuffleSplit

# === Paths ===
results_path = os.path.join("..", "results")
model_path = os.path.join(results_path, "xgb_rfe_tuned_model_better.joblib")
data_path = os.path.join(results_path, "combined_18fts.npz")

# === Load saved XGB model ===
model_dict = joblib.load(model_path)
best_model = model_dict["model"]
threshold = model_dict["threshold"]
selected_features = model_dict["selected_features"]

# === Load data ===
data = np.load(data_path)
X, y, groups = data["features"], data["labels"], data["subject_ids"]

# === Map feature names to indices and select features ===
channels = ["F4A1", "C4A1", "O2A1"]
base_feats = ["delta", "theta", "alpha", "beta", "gamma", "δ/θ", "δ/α", "α/β", "θ/α",
              "activity", "mobility", "complexity", "mean", "std", "skew"]
feature_names = [f"{ch}_{f}" for ch in channels for f in base_feats]

selected_idx = [feature_names.index(f) for f in selected_features]
X = X[:, selected_idx]

# === Use same train/test split as hypertuning ===
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
_, test_idx = next(gss.split(X, y, groups))
X_eval, y_eval = X[test_idx], y[test_idx]

# === Predict using saved threshold ===
y_probs = best_model.predict_proba(X_eval)[:, 1]
y_pred = (y_probs >= threshold).astype(int)

# === Evaluate ===
print("Selected Features:", selected_features)
print("\nConfusion Matrix:\n", confusion_matrix(y_eval, y_pred))
print("\nClassification Report:\n", classification_report(y_eval, y_pred))
accuracy = accuracy_score(y_eval, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")