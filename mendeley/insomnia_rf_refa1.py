import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# === Load features, labels, and subject IDs ===
data = np.load("../../results/mendeley_8fts.npz")
X = data['features']
y = data['labels']
groups = data['subject_ids']  # <- prevents leakage

# === Group-aware train/test split ===
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# === Train Random Forest ===
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    class_weight='balanced'
)
clf.fit(X_train, y_train)

# === Evaluate ===
y_pred = clf.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# === Evaluate again (if not already done) ===
y_pred = clf.predict(X_test)

# === Confusion matrix ===
cm = confusion_matrix(y_test, y_pred)
labels = ["Normal", "Insomnia"]

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix (Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%)")
plt.tight_layout()
plt.show()

# === Classification report ===
report = classification_report(y_test, y_pred, output_dict=True)
metrics = ["precision", "recall", "f1-score"]
x = ["Normal", "Insomnia"]
x_pos = np.arange(len(x))

plt.figure(figsize=(10, 6))
for i, metric in enumerate(metrics):
    plt.bar(x_pos + i*0.25, [report[label][metric] for label in ["0", "1"]],
            width=0.25, label=metric.capitalize())

plt.xticks(x_pos + 0.25, x)
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Classification Report Metrics")
plt.legend()
plt.tight_layout()
plt.show()