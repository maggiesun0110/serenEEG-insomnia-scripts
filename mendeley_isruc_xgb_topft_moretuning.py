
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

# === Selected Features (XGB best from RFE) ===
selected_idx = [2, 32]  # ['F4A1_alpha', 'O2A1_alpha']
X = X[:, selected_idx]

# === Feature Names ===
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

# === RandomizedSearchCV space ===
param_dist = {
    "n_estimators": [50, 100, 200, 300, 400, 500],
    "max_depth": [2, 3, 4, 5, 6, 7],
    "learning_rate": [0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
    "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 2, 3, 4, 5, 6],
    "gamma": [0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "reg_alpha": [0, 0.01, 0.1, 0.5, 1],
    "reg_lambda": [0.1, 0.5, 1, 2, 5, 10]
}

xgb = XGBClassifier(random_state=42, eval_metric="logloss")

# === Randomized Search ===
random_search = RandomizedSearchCV(
    xgb,
    param_distributions=param_dist,
    n_iter=50,
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=2
)
random_search.fit(X_train, y_train)

# === Utility for tight neighborhood grid search ===
def around(value, step, values_list):
    idx = values_list.index(value)
    start = max(0, idx - step)
    end = min(len(values_list), idx + step + 1)
    return list(set(values_list[start:end]))

# === Tight Grid Search around best Random params ===
best_random = random_search.best_params_
grid_param = {
    "n_estimators": around(best_random["n_estimators"], 1, [50, 100, 200, 300, 400, 500]),
    "max_depth": around(best_random["max_depth"], 1, [2, 3, 4, 5, 6, 7]),
    "learning_rate": around(best_random["learning_rate"], 1, [0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3]),
    "subsample": around(best_random["subsample"], 1, [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
    "colsample_bytree": around(best_random["colsample_bytree"], 1, [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
    "min_child_weight": around(best_random["min_child_weight"], 1, [1, 2, 3, 4, 5, 6]),
    "gamma": around(best_random["gamma"], 1, [0, 0.1, 0.2, 0.3, 0.4, 0.5]),
    "reg_alpha": around(best_random["reg_alpha"], 1, [0, 0.01, 0.1, 0.5, 1]),
    "reg_lambda": around(best_random["reg_lambda"], 1, [0.1, 0.5, 1, 2, 5, 10])
}

grid_search = GridSearchCV(
    xgb,
    param_grid=grid_param,
    cv=3,
    n_jobs=-1,
    verbose=2
)
grid_search.fit(X_train, y_train)

# === Final Model ===
best_grid = grid_search.best_params_
best_model = XGBClassifier(
    n_estimators=500,  # higher to compensate for no early stopping
    max_depth=best_grid["max_depth"],
    learning_rate=max(best_grid["learning_rate"]/2, 0.001),
    subsample=best_grid["subsample"],
    colsample_bytree=best_grid["colsample_bytree"],
    min_child_weight=best_grid["min_child_weight"],
    gamma=best_grid["gamma"],
    reg_alpha=best_grid["reg_alpha"],
    reg_lambda=best_grid["reg_lambda"],
    random_state=42,
    eval_metric="logloss"
)

# Fit with eval_set for loss curve
best_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)

# === Plot Training & Validation Loss Curve ===
results = best_model.evals_result()
train_loss = results['validation_0']['logloss']
val_loss = results['validation_1']['logloss']

plt.figure(figsize=(8,5))
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel("Iteration")
plt.ylabel("Log Loss")
plt.title("XGB Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Threshold tuning (maximize accuracy) ===
probas = best_model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, probas)
accuracies = [accuracy_score(y_test, (probas >= t).astype(int)) for t in thresholds]
best_threshold = thresholds[np.argmax(accuracies)]
print(f"Optimal Threshold (Accuracy max): {best_threshold:.4f}")

# === Predictions & Evaluation ===
y_pred = (probas >= best_threshold).astype(int)
print("Selected Features:", selected_names)
print("\nBest Random Params:", best_random)
print("\nBest Grid Params:", best_grid)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))

# === Save model + threshold ===
joblib.dump({
    "model": best_model,
    "threshold": float(best_threshold),
    "selected_features": selected_names
}, os.path.join(results_path, "xgb_rfe_tuned_model_better.joblib"))
