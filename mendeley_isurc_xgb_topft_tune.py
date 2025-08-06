import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_curve

# === Paths ===
results_path = os.path.join("..", "results")
data = np.load(os.path.join(results_path, "combined_18fts.npz"))
X, y, groups = data["features"], data["labels"], data["subject_ids"]

# === Selected Features (XGB best from RFE)
selected_idx = [2, 32]  # ['F4A1_alpha', 'O2A1_alpha']
X = X[:, selected_idx]

# === Feature Names
channels = ["F4A1", "C4A1", "O2A1"]
base_feats = ["delta", "theta", "alpha", "beta", "gamma", "δ/θ", "δ/α", "α/β", "θ/α", "activity", "mobility", "complexity", "mean", "std", "skew"]
feature_names = [f"{ch}_{f}" for ch in channels for f in base_feats]
selected_names = [feature_names[i] for i in selected_idx]

# === Group-aware split
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# === RandomizedSearchCV
param_dist = {
    "n_estimators": [50, 100, 200],
    "max_depth": [2, 3, 4, 5],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0]
}

xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
random_search = RandomizedSearchCV(xgb, param_distributions=param_dist, n_iter=20, cv=3, random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)

# === GridSearchCV (refined)
best_random = random_search.best_params_
grid_param = {
    "n_estimators": [best_random["n_estimators"]],
    "max_depth": [best_random["max_depth"]],
    "learning_rate": [best_random["learning_rate"]],
    "subsample": [best_random["subsample"]],
    "colsample_bytree": [best_random["colsample_bytree"]]
}
grid_search = GridSearchCV(xgb, param_grid=grid_param, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)

# === Final Model + Manual Threshold
best_model = grid_search.best_estimator_
probas = best_model.predict_proba(X_test)[:, 1]
threshold = 0.90
y_pred = (probas >= threshold).astype(int)

# === Evaluation
print("Selected Features:", selected_names)
print("Threshold:", threshold)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))

# === Save model + threshold
joblib.dump({
    "model": best_model,
    "threshold": threshold,
    "selected_features": selected_names
}, os.path.join(results_path, "xgb_rfe_thresh70_model.joblib"))

# === Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test, probas)
plt.plot(thresholds, precision[:-1], label='Precision')
plt.plot(thresholds, recall[:-1], label='Recall')
plt.axvline(x=threshold, color='red', linestyle='--', label='Chosen Threshold (0.90)')
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision & Recall vs Threshold (XGB)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_path, "xgb_threshold_curve.png"))
plt.show()