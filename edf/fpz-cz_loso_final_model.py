import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter
import warnings

# === Load Data ===
data_path = "../results/features_with_ids.npz"
assert os.path.exists(data_path), f"❌ Feature file not found: {data_path}"

data = np.load(data_path, allow_pickle=True)
X = data["X"]
y = data["y"]
subject_ids = data["subject_ids"]

# === Sanity Checks ===
assert len(X) == len(y) == len(subject_ids), "❌ Mismatched X, y, subject_ids lengths"
assert set(np.unique(y)) <= {0, 1}, f"❌ Unexpected labels found: {set(np.unique(y))}"
if len(np.unique(subject_ids)) < len(subject_ids):
    print("✅ Multiple samples per subject confirmed.")

# === Transparency ===
print("📊 Dataset Overview:")
print(f"  Total samples: {len(X)}")
print(f"  Label distribution: {Counter(y)}")
print(f"  Unique subjects: {len(np.unique(subject_ids))}")
print(f"  Subject sample counts: {dict(Counter(subject_ids))}")

# === Subject-Leakage-Free Stratified CV ===
print("\n🔐 Performing 5-fold StratifiedGroupKFold (no subject leakage)...")
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

cv_accuracies = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups=subject_ids)):
    train_subjects = set(subject_ids[train_idx])
    val_subjects = set(subject_ids[val_idx])
    overlap = train_subjects.intersection(val_subjects)
    assert not overlap, f"Fold {fold+1}: Subject leakage detected! Overlap: {overlap}"

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    cv_accuracies.append(acc)
    print(f"  📌 Fold {fold + 1} Accuracy: {acc:.4f}")

# === Final Report ===
print(f"\n✅ Mean CV Accuracy: {np.mean(cv_accuracies) * 100:.2f}% ± {np.std(cv_accuracies) * 100:.2f}%")

# === Final Training on All Data ===
print("\n🚀 Training final model on ALL data (for external testing)...")
final_model = RandomForestClassifier(n_estimators=100, random_state=42)
final_model.fit(X, y)

# === Save Final Model ===
model_save_path = "../results/final_insomnia_rf_model.pkl"
joblib.dump(final_model, model_save_path)
print(f"💾 Final model saved to: {model_save_path}")