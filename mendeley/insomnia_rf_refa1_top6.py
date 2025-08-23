import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# === Load 6-feature RFE-extracted data ===
data = np.load("../../results/mendeley_rfe6.npz")
X = data['features']
y = data['labels']
groups = data['subject_ids']

# === Group-aware split (prevent subject-level leakage) ===
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# === Train RF ===
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    class_weight='balanced',
    random_state=42
)
clf.fit(X_train, y_train)

# === Predict + Evaluate ===
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print(f"Accuracy: {acc * 100:.2f}%")

# === Plot Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
labels = ["Normal", "Insomnia"]

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix (Accuracy: {acc*100:.2f}%)")
plt.tight_layout()
plt.show()

# === Plot classification metrics (precision, recall, f1) ===
report = classification_report(y_test, y_pred, output_dict=True)
metrics = ["precision", "recall", "f1-score"]
x_labels = ["Normal", "Insomnia"]
x_pos = np.arange(len(x_labels))

plt.figure(figsize=(10, 6))
for i, metric in enumerate(metrics):
    plt.bar(x_pos + i * 0.25,
            [report[label][metric] for label in ["0", "1"]],
            width=0.25, label=metric.capitalize())

plt.xticks(x_pos + 0.25, x_labels)
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Classification Report Metrics (6 RFE Features)")
plt.legend()
plt.tight_layout()
plt.show()