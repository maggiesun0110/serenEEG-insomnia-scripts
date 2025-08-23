import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt

# === Load data ===
data = np.load("../../results/mendeley_12fts.npz")
X = data['features']
y = data['labels']
groups = data['subject_ids']

# === Group-aware split ===
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# === Base RF model ===
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    class_weight='balanced',
    random_state=42
)

# === RFE selector ===
# Select top 6 features (you can change n_features_to_select as needed)
rfe = RFE(estimator=clf, n_features_to_select=6)
rfe.fit(X_train, y_train)

print("RFE Selected Features Mask:", rfe.support_)
print("RFE Feature Ranking:", rfe.ranking_)

# === Train RF on selected features only ===
clf.fit(X_train[:, rfe.support_], y_train)

# === Predict and Evaluate ===
y_pred = clf.predict(X_test[:, rfe.support_])

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))

# === Plot feature importance for selected features ===
feature_names = [
    "F4A1_mobility", "F4A1_alpha_beta",
    "C4A1_mobility", "C4A1_alpha_beta",
    "O2A1_mobility", "O2A1_alpha_beta", "O2A1_mean", "O2A1_std", "O2A1_complexity"
]

selected_features = np.array(feature_names)[rfe.support_]
importances = clf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

plt.figure(figsize=(8, 4))
plt.barh(selected_features[sorted_idx], importances[sorted_idx], color="teal")
plt.xlabel("Feature Importance")
plt.title("Feature Importances (RFE-selected features)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

import joblib

# Assume clf is your trained RandomForestClassifier on the 12 features (but you only select the RFE top 6 later)
# Save the model to disk
joblib.dump(clf, '../../results/insomnia_rf_model_rfe6.pkl')
print("Model saved to ../../results/insomnia_rf_model_rfe6.pkl")