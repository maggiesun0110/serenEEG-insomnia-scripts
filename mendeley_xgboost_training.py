import os
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report

# === Config ===
data_path = "../results/features_a1_advanced_all_with_ids.npz"
model_save_path = "../results/xgb_a1_model_clean.joblib"
test_size = 0.2
random_state = 42

assert os.path.exists(data_path), f"❌ Data file not found: {data_path}"

# === Load Data ===
data = np.load(data_path, allow_pickle=True)
X = data["X"]
y = data["y"]
subject_ids = data["subject_ids"]

print(f"✅ Loaded {X.shape[0]} samples with {X.shape[1]} features.")
print(f"✅ Found {len(np.unique(subject_ids))} unique subjects.")

# === Group-aware split to prevent leakage ===
gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
train_idx, test_idx = next(gss.split(X, y, groups=subject_ids))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"🧪 Train set: {X_train.shape[0]} samples")
print(f"🧪 Test set: {X_test.shape[0]} samples")

# === Train XGBoost ===
model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=random_state,
)

print("\n🔨 Training model...")
model.fit(X_train, y_train)

# === Evaluate on Test Set ===
print("\n📈 Evaluating on held-out test set...")
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"🎯 Test Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred, digits=4))

# === Save model ===
joblib.dump(model, model_save_path)
print(f"💾 Model saved to: {model_save_path}")