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