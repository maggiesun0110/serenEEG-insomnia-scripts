import os
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report

# === Config ===
data_path = "../results/features_a1_advanced_all_with_ids.npz"
model_save_path = "../results/lgb_a1_model_boosted.txt"
test_size = 0.2  # 80/20 train/test split
val_size = 0.2   # 20% of train for validation
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

# === Further split train into train/validation with stratification ===
X_train2, X_val, y_train2, y_val = train_test_split(
    X_train, y_train, test_size=val_size, stratify=y_train, random_state=random_state
)

print(f"📊 Training on {X_train2.shape[0]} samples, validating on {X_val.shape[0]} samples")

# === Create LightGBM datasets ===
train_data = lgb.Dataset(X_train2, label=y_train2)
val_data = lgb.Dataset(X_val, label=y_val)

# === LightGBM parameters ===
params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'seed': random_state,
    'verbose': -1,
}

print("\n🚀 Training LightGBM with early stopping...")

model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[val_data],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=50)  # prints every 50 rounds
    ],
)

print("\n🎯 Evaluating on held-out test set...")

y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
print(f"LightGBM Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred))

# === Save model ===
model.save_model(model_save_path)
print(f"💾 Model saved as {model_save_path}")