import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils import resample
import joblib

# === Set Random Seeds for Reproducibility ===
random.seed(42)
np.random.seed(42)

# === Load Features, Labels, and Subject IDs ===
feature_file = "../results/features_a1_advanced_all_with_ids.npz"
model_output_path = "../results/rf_a1_model.joblib"

data = np.load(feature_file)
X = data["X"]
y = data["y"]
subject_ids = data["subject_ids"]

print(f"✅ Loaded {X.shape[0]} samples with {X.shape[1]} features each.")
print(f"📌 Found {len(np.unique(subject_ids))} unique subjects.")

# === Class Balancing by Upsampling Minority Class ===
combined = list(zip(X, y, subject_ids))
class0 = [item for item in combined if item[1] == 0]
class1 = [item for item in combined if item[1] == 1]

# Identify majority and minority classes
if len(class0) > len(class1):
    majority = class0
    minority = class1
else:
    majority = class1
    minority = class0

# Upsample minority class with replacement
minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)

balanced = majority + minority_upsampled
np.random.shuffle(balanced)

X_bal = np.array([item[0] for item in balanced])
y_bal = np.array([item[1] for item in balanced])
ids_bal = np.array([item[2] for item in balanced])

print(f"⚖️ Balanced dataset: {len(X_bal)} total samples (upsampled minority to {len(majority)})")

# === Group-aware Train/Test Split ===
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X_bal, y_bal, groups=ids_bal))

X_train, X_test = X_bal[train_idx], X_bal[test_idx]
y_train, y_test = y_bal[train_idx], y_bal[test_idx]

# === Train Initial Random Forest for Feature Importance ===
clf_initial = RandomForestClassifier(n_estimators=100, random_state=42)
clf_initial.fit(X_train, y_train)

importances = clf_initial.feature_importances_
top_k = 15  # select top 15 features
indices = np.argsort(importances)[::-1][:top_k]

print(f"\n🌟 Top {top_k} features based on importance:")
for i, idx in enumerate(indices):
    print(f"{i+1:2d}. Feature {idx:3d} → Importance: {importances[idx]:.4f}")

# === Reduce features to top_k ===
X_train_reduced = X_train[:, indices]
X_test_reduced = X_test[:, indices]

# === Retrain Random Forest with selected features ===
clf_final = RandomForestClassifier(n_estimators=100, random_state=42)
clf_final.fit(X_train_reduced, y_train)

# === Evaluate ===
y_pred = clf_final.predict(X_test_reduced)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy with top {top_k} features: {acc:.4f}")
print(classification_report(y_test, y_pred, target_names=["Normal", "Insomnia"]))

# === Save final model ===
joblib.dump(clf_final, model_output_path)
print(f"💾 Model saved to {model_output_path}")