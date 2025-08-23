import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, cohen_kappa_score, roc_curve
)
from sklearn.model_selection import GroupShuffleSplit

# === Load data ===
results_path = os.path.join("..", "results")
data = np.load(os.path.join(results_path, "combined_18fts.npz"))
X, y, groups = data["features"], data["labels"], data["subject_ids"]

# === Selected features ===
selected_idx = [2, 7, 17, 32, 37]
X = X[:, selected_idx]

# === Train/test split (group-aware) ===
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# === Learning curve setup with final hyperparameters ===
rf_curve = RandomForestClassifier(
    n_estimators=10,  # start small
    warm_start=True,
    max_depth=20,
    min_samples_split=7,
    min_samples_leaf=6,
    max_features='sqrt',
    bootstrap=False,
    class_weight='balanced',
    random_state=42
)

# Storage for metrics
train_acc, val_acc = [], []
train_auc, val_auc = [], []
train_kappa, val_kappa = [], []

n_trees_list = list(range(10, 301, 10))  # up to final n_estimators=300
threshold = 0.7

for n in n_trees_list:
    rf_curve.set_params(n_estimators=n)
    rf_curve.fit(X_train, y_train)
    
    # Probabilities
    train_probs = rf_curve.predict_proba(X_train)[:, 1]
    val_probs = rf_curve.predict_proba(X_test)[:, 1]
    
    # Thresholded predictions
    y_train_pred = (train_probs >= threshold).astype(int)
    y_val_pred = (val_probs >= threshold).astype(int)
    
    # Accuracy
    train_acc.append(accuracy_score(y_train, y_train_pred))
    val_acc.append(accuracy_score(y_test, y_val_pred))
    
    # AUC
    train_auc.append(roc_auc_score(y_train, train_probs))
    val_auc.append(roc_auc_score(y_test, val_probs))
    
    # Cohen's Kappa
    train_kappa.append(cohen_kappa_score(y_train, y_train_pred))
    val_kappa.append(cohen_kappa_score(y_test, y_val_pred))

# === Plot Accuracy & AUC ===
plt.figure(figsize=(10, 6))
plt.plot(n_trees_list, train_acc, label='Train Accuracy', marker='o')
plt.plot(n_trees_list, val_acc, label='Validation Accuracy', marker='o')
plt.plot(n_trees_list, train_auc, label='Train AUC', linestyle='--', marker='x')
plt.plot(n_trees_list, val_auc, label='Validation AUC', linestyle='--', marker='x')
plt.xlabel("Number of Trees")
plt.ylabel("Score")
plt.title("Random Forest Learning Curve with Accuracy & AUC (Group-Aware)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Plot Cohen's Kappa ===
plt.figure(figsize=(8, 5))
plt.plot(n_trees_list, train_kappa, label="Train Kappa", marker='o')
plt.plot(n_trees_list, val_kappa, label="Validation Kappa", marker='o')
plt.xlabel("Number of Trees")
plt.ylabel("Cohen's Kappa")
plt.title("Cohen's Kappa over Trees (Group-Aware)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Final ROC curve for last model ===
fpr, tpr, _ = roc_curve(y_test, val_probs)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {val_auc[-1]:.3f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Final Model ROC Curve (Group-Aware)")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()