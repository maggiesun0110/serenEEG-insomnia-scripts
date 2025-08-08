import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# === Paths ===
results_path = os.path.join("..", "results")
data = np.load(os.path.join(results_path, "combined_18fts.npz"))
X, y, groups = data["features"], data["labels"], data["subject_ids"]

# === Load saved model bundle ===
model_bundle = joblib.load(os.path.join(results_path, "rf_rfe_tuned_model_better.joblib"))
best_model = model_bundle["model"]
threshold = model_bundle["threshold"]
selected_idx = model_bundle["selected_idx"]

# === Use selected features ===
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

        # Retrain model on training fold
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

# Plotting results over repeats
plt.figure(figsize=(12, 7))
plt.plot(range(1, n_repeats + 1), all_acc, marker='o', label='Accuracy')
plt.plot(range(1, n_repeats + 1), all_prec, marker='o', label='Precision')
plt.plot(range(1, n_repeats + 1), all_rec, marker='o', label='Recall')
plt.plot(range(1, n_repeats + 1), all_f1, marker='o', label='F1-score')

plt.title(f"Repeated {n_splits}-Fold GroupKFold CV Metrics over {n_repeats} Runs")
plt.xlabel("Repeat Number")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(range(1, n_repeats + 1))
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()