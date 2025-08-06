import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# === Load data ===
data = np.load("../../results/mendeley_12fts.npz")
X = data['features']
y = data['labels']
groups = data['subject_ids']

# === RFE-selected feature indices (adjust if needed) ===
rfe_mask = np.array([True, True, True, False, False, True, False, True, True])  
X_rfe = X[:, rfe_mask]

# === Group-aware train/test split ===
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X_rfe, y, groups))
X_train, X_test = X_rfe[train_idx], X_rfe[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# === Initialize XGBoost classifier ===
xgb_clf = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()  # handle imbalance
)

# === Train ===
xgb_clf.fit(X_train, y_train)

# === Predict and evaluate ===
y_pred = xgb_clf.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))

# === Plot confusion matrix ===
labels = ["Normal", "Insomnia"]
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix (Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%)")
plt.tight_layout()
plt.show()

import joblib

joblib.dump(xgb_clf, '../../results/insomnia_xgb_model_rfe6.pkl')
print("Model saved to ../../results/insomnia_xgb_model_rfe6.pkl")