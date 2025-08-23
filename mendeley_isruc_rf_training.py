import numpy as np
import os

# === Paths ===
results_path = os.path.join("..", "results")
mendeley_path = os.path.join(results_path, "mendeley_12fts.npz")
isruc_path = os.path.join(results_path, "isruc_9features.npz")
combined_path = os.path.join(results_path, "combined_9features.npz")

# === Load both ===
m = np.load(mendeley_path)
i = np.load(isruc_path)

X_combined = np.vstack([m["features"], i["features"]])
y_combined = np.hstack([m["labels"], i["labels"]])
subject_ids_combined = np.hstack([m["subject_ids"], i["subject_ids"]])

# === Save combined ===
np.savez(combined_path, features=X_combined, labels=y_combined, subject_ids=subject_ids_combined)
print("Combined shape:", X_combined.shape, y_combined.shape)

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import GroupShuffleSplit, cross_val_score
import joblib

# === Load combined dataset ===
data = np.load("../results/combined_9features.npz")
X, y, groups = data["features"], data["labels"], data["subject_ids"]

# === Base model ===
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# === RFE to select top 6 features ===
rfe = RFE(estimator=rf, n_features_to_select=5)
X_selected = rfe.fit_transform(X, y)
print("Selected feature indices:", np.where(rfe.support_)[0])

# === Cross-validate with group split ===
gss = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
scores = cross_val_score(rf, X_selected, y, groups=groups, cv=gss)

print("CV Accuracy scores:", scores)
print("Mean accuracy:", scores.mean())

rf.fit(X_selected, y)
joblib.dump(rf, "../results/rf_rfe_model.joblib")