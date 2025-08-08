# === LOGO evaluation of already hypertuned RF model ===
import os
import joblib
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# === Paths ===
results_path = os.path.join("..", "results")
model_path = os.path.join(results_path, "rf_rfe_tuned_model_better.joblib")

# === Load saved model dict ===
saved_data = joblib.load(model_path)
model = saved_data["model"]
threshold = saved_data["threshold"]
selected_idx = saved_data["selected_idx"]

print(f"Loaded model from: {model_path}")
print(f"Using threshold: {threshold}")
print(f"Selected feature indices: {selected_idx}")

# === Load dataset ===
data = np.load(os.path.join(results_path, "combined_18fts.npz"))
X, y, groups = data["features"], data["labels"], data["subject_ids"]

# Apply same feature selection
X = X[:, selected_idx]

# === Leave-One-Group-Out CV ===
logo = LeaveOneGroupOut()

all_y_true = []
all_y_pred = []

fold_num = 1
for train_idx, test_idx in logo.split(X, y, groups):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Fit model on training fold
    model.fit(X_train, y_train)

    # Predict probabilities & apply stored threshold
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)

    # Store for overall metrics
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)

    # Fold metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n=== Fold {fold_num} (Group left out: {np.unique(groups[test_idx])}) ===")
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

    fold_num += 1

# === Overall metrics ===
print("\n=== Overall Leave-One-Group-Out Performance ===")
print("Confusion Matrix:\n", confusion_matrix(all_y_true, all_y_pred))
print("\nClassification Report:\n", classification_report(all_y_true, all_y_pred))
print("Accuracy: {:.2f}%".format(accuracy_score(all_y_true, all_y_pred) * 100))