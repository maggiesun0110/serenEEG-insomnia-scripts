import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# === Paths ===
results_path = os.path.join("..", "results")
data = np.load(os.path.join(results_path, "combined_18fts.npz"))
X, y, groups = data["features"], data["labels"], data["subject_ids"]

# === Load saved XGB model bundle ===
model_bundle = joblib.load(os.path.join(results_path, "xgb_rfe_tuned_model_better.joblib"))
best_model = model_bundle["model"]
threshold = model_bundle["threshold"]
selected_features = model_bundle["selected_features"]

# === Map feature names back to indices ===
channels = ["F4A1", "C4A1", "O2A1"]
base_feats = ["delta", "theta", "alpha", "beta", "gamma", "δ/θ", "δ/α", "α/β", "θ/α",
              "activity", "mobility", "complexity", "mean", "std", "skew"]
feature_names = [f"{ch}_{f}" for ch in channels for f in base_feats]
selected_idx = [feature_names.index(f) for f in selected_features]
X = X[:, selected_idx]

# Settings
n_splits = 5
n_repeats = 10

# Storage for all repeats results
all_acc = []
all_prec = []
all_rec = []
all_f1 = []

print(f"Running {n_repeats} runs of {n_splits}-Fold GroupKFold CV with threshold {threshold}\n")

for repeat in range(1, n_repeats + 1):
    print(f"=== Repeat {repeat} ===")
    # Shuffle groups for each repeat to get different splits
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed=repeat)
    shuffled_groups = rng.permutation(unique_groups)
    
    # Map original groups to shuffled order
    group_map = {g: i for i, g in enumerate(shuffled_groups)}
    shuffled_group_ids = np.array([group_map[g] for g in groups])
    
    gkf = GroupKFold(n_splits=n_splits)
    
    fold_acc = []
    fold_prec = []
    fold_rec = []
    fold_f1 = []
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, shuffled_group_ids), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Retrain XGB model on training fold
        best_model.fit(X_train, y_train)

        probs = best_model.predict_proba(X_test)[:, 1]
        y_pred = (probs >= threshold).astype(int)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f" Fold {fold}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
        
        fold_acc.append(acc)
        fold_prec.append(prec)
        fold_rec.append(rec)
        fold_f1.append(f1)

    # Average over folds for this repeat
    mean_acc = np.mean(fold_acc)
    mean_prec = np.mean(fold_prec)
    mean_rec = np.mean(fold_rec)
    mean_f1 = np.mean(fold_f1)
    
    print(f" Repeat {repeat} averages: Accuracy={mean_acc:.4f}, Precision={mean_prec:.4f}, Recall={mean_rec:.4f}, F1={mean_f1:.4f}\n")
    
    all_acc.append(mean_acc)
    all_prec.append(mean_prec)
    all_rec.append(mean_rec)
    all_f1.append(mean_f1)

# === Overall metrics across all repeats ===
overall_acc = np.mean(all_acc)
overall_prec = np.mean(all_prec)
overall_rec = np.mean(all_rec)
overall_f1 = np.mean(all_f1)

print("="*50)
print(f"Overall Accuracy over {n_repeats}×{n_splits} CV: {overall_acc:.4f}")
print(f"Overall Precision: {overall_prec:.4f}")
print(f"Overall Recall:    {overall_rec:.4f}")
print(f"Overall F1-score:  {overall_f1:.4f}")
print("="*50)

# Plotting results over repeats
plt.figure(figsize=(12, 7))
plt.plot(range(1, n_repeats + 1), all_acc, marker='o', label='Accuracy')
plt.plot(range(1, n_repeats + 1), all_prec, marker='o', label='Precision')
plt.plot(range(1, n_repeats + 1), all_rec, marker='o', label='Recall')
plt.plot(range(1, n_repeats + 1), all_f1, marker='o', label='F1-score')

plt.title(f"Repeated {n_splits}-Fold GroupKFold CV Metrics over {n_repeats} Runs (XGB)")
plt.xlabel("Repeat Number")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(range(1, n_repeats + 1))
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()