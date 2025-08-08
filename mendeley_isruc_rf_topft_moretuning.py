import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# === Paths ===
results_path = os.path.join("..", "results")
data = np.load(os.path.join(results_path, "combined_18fts.npz"))
X, y, groups = data["features"], data["labels"], data["subject_ids"]

# === Selected Features (RF best from RFE) ===
selected_idx = [2, 7, 17, 32, 37]
X = X[:, selected_idx]

# === Feature names (optional) ===
channels = ["F4A1", "C4A1", "O2A1"]
base_feats = ["delta", "theta", "alpha", "beta", "gamma", "δ/θ", "δ/α", "α/β", "θ/α",
              "activity", "mobility", "complexity", "mean", "std", "skew"]
feature_names = [f"{ch}_{f}" for ch in channels for f in base_feats]
selected_names = [feature_names[i] for i in selected_idx]

# === Group-aware split ===
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# === Random Forest with balanced class weights ===
rf = RandomForestClassifier(random_state=42, class_weight='balanced')

# === Widened RandomizedSearchCV param distribution ===
param_dist = {
    "n_estimators": [200, 400, 600, 800, 1000],
    "max_depth": [None, 20, 30, 40, 50, 70],
    "min_samples_split": [2, 3, 4, 6, 8, 10],
    "min_samples_leaf": [1, 2, 3, 4, 5],
    "bootstrap": [True, False],
    "max_features": ['sqrt', 'log2', None]
}

random_search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=75,
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=2,
    scoring='f1'
)
random_search.fit(X_train, y_train)

# === Narrow GridSearchCV around best random params ===
best_random = random_search.best_params_

grid_param = {
    "n_estimators": [
        best_random["n_estimators"],
        min(best_random["n_estimators"] + 100, 1000)
    ],
    "max_depth": (
        [best_random["max_depth"], None]
        if best_random["max_depth"] is not None
        else [None, 30, 40]
    ),
    "min_samples_split": [
        max(2, best_random["min_samples_split"] - 1),
        best_random["min_samples_split"],
        best_random["min_samples_split"] + 1
    ],
    "min_samples_leaf": [
        max(1, best_random["min_samples_leaf"] - 1),
        best_random["min_samples_leaf"],
        best_random["min_samples_leaf"] + 1
    ],
    "bootstrap": [best_random["bootstrap"]],
    "max_features": [best_random["max_features"]]
}

grid_search = GridSearchCV(
    rf,
    param_grid=grid_param,
    cv=3,
    n_jobs=-1,
    verbose=2,
    scoring='f1'
)
grid_search.fit(X_train, y_train)

# === Final Evaluation with fixed threshold ===
best_model = grid_search.best_estimator_
probs = best_model.predict_proba(X_test)[:, 1]
threshold = 0.70
y_pred = (probs >= threshold).astype(int)

print("\n=== Best Model Hyperparameters ===")
for k, v in grid_search.best_params_.items():
    print(f"{k}: {v}")

print("Selected Features:", selected_names)
print(f"\n--- Final Evaluation (Threshold = {threshold}) ---")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))

# === Save final tuned model ===
joblib.dump({
    "model": best_model,
    "threshold": threshold,
    "selected_idx": selected_idx  # also save selected features indices for convenience
}, os.path.join(results_path, "rf_rfe_tuned_model_better.joblib"))

# === Threshold tuning block ===
print("\n--- Threshold Tuning ---")
for t in np.arange(0.1, 1.0, 0.1):
    y_thresh = (probs >= t).astype(int)
    acc = accuracy_score(y_test, y_thresh)
    prec = precision_score(y_test, y_thresh, zero_division=0)
    rec = recall_score(y_test, y_thresh, zero_division=0)
    f1 = f1_score(y_test, y_thresh, zero_division=0)
    print(f"Threshold {t:.2f} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")

# === Plot accuracy across RandomizedSearchCV ===
results_df = pd.DataFrame(random_search.cv_results_)
plt.figure(figsize=(10, 6))
plt.plot(results_df['mean_test_score'], marker='o', linestyle='-', label='Mean CV F1 Score')
plt.axhline(y=results_df['mean_test_score'].max(), color='red', linestyle='--', label='Best F1 Score')
plt.title("RandomizedSearchCV F1 Scores (Random Forest)")
plt.xlabel("Search Iteration")
plt.ylabel("Mean CV F1 Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_path, "rf_random_search_f1_plot_balanced.png"))
plt.show()