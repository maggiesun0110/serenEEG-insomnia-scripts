import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter
import joblib
import os

# === Load your data ===
data_path = "../results/features_with_ids.npz"
assert os.path.exists(data_path), f"File not found: {data_path}"

data = np.load(data_path, allow_pickle=True)
X = data['X']
y = data['y']
subject_ids = data['subject_ids']

print(f"Total samples: {len(X)}")
print(f"Label distribution: {Counter(y)}")
print(f"Unique subjects: {len(np.unique(subject_ids))}")

# === LOSO Cross-Validation ===
logo = LeaveOneGroupOut()

accuracies = []
all_y_true = []
all_y_pred = []
test_subjects_list = []

for train_idx, test_idx in logo.split(X, y, groups=subject_ids):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)
    test_subjects_list.append(subject_ids[test_idx][0])

    print(f"Test subject: {subject_ids[test_idx][0]} - Accuracy: {acc:.4f}")

# === LOSO Metrics ===
average_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)

print(f"\nAverage LOSO accuracy: {average_accuracy:.4f}")
print(f"Standard deviation: {std_accuracy:.4f}\n")
print("Classification report for all test folds combined:")
print(classification_report(all_y_true, all_y_pred))

# === Save LOSO results ===
loso_save_path = "../results/loso_results.npz"
np.savez(
    loso_save_path,
    all_y_true=np.array(all_y_true),
    all_y_pred=np.array(all_y_pred),
    test_subjects=np.array(test_subjects_list),
    accuracies=np.array(accuracies),
    average_accuracy=average_accuracy,
)
print(f"LOSO results saved to {loso_save_path}")
