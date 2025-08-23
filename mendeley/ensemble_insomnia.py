import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

# Load data and subject IDs
data = np.load("../results/features_a1_advanced_all_with_ids.npz")
X = data["X"]
y = data["y"]
subject_ids = data["subject_ids"]

# Group-aware train/test split to prevent subject leakage
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=subject_ids))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# RF feature selection indices (top 15)
rf_feature_indices = [12, 4, 3, 2, 6, 15, 9, 16, 10, 7, 13, 1, 8, 5, 0]
X_train_rf = X_train[:, rf_feature_indices]
X_test_rf = X_test[:, rf_feature_indices]

# Prepare placeholders for out-of-fold predictions
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize OOF prediction arrays: shape = (train_samples, 2 classes)
rf_oof = np.zeros((X_train.shape[0], 2))
xgb_oof = np.zeros((X_train.shape[0], 2))
svm_oof = np.zeros((X_train.shape[0], 2))

print("🚀 Generating out-of-fold predictions for base models...")

for fold, (train_fold_idx, val_fold_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"Training fold {fold + 1} / {n_splits}...")

    # Split fold data
    X_tr, X_val = X_train[train_fold_idx], X_train[val_fold_idx]
    y_tr, y_val = y_train[train_fold_idx], y_train[val_fold_idx]

    # RF train/val (with selected features)
    X_tr_rf, X_val_rf = X_tr[:, rf_feature_indices], X_val[:, rf_feature_indices]
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_tr_rf, y_tr)
    rf_oof[val_fold_idx] = rf.predict_proba(X_val_rf)

    # XGB train/val (all features)
    xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05,
                        use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb.fit(X_tr, y_tr)
    xgb_oof[val_fold_idx] = xgb.predict_proba(X_val)

    # SVM train/val (all features)
    svm = SVC(kernel='rbf', C=100, gamma=0.01, class_weight='balanced', probability=True, random_state=42)
    svm.fit(X_tr, y_tr)
    svm_oof[val_fold_idx] = svm.predict_proba(X_val)

# Stack OOF predictions as features for meta-model
meta_X_train = np.hstack([rf_oof, xgb_oof, svm_oof])
meta_y_train = y_train

print("\nTraining meta-model on stacked features...")
meta_model = LogisticRegression(random_state=42, max_iter=1000)
meta_model.fit(meta_X_train, meta_y_train)

# Train base models on full training data for test prediction
print("\nTraining base models on full training data...")
rf_final = RandomForestClassifier(n_estimators=100, random_state=42)
rf_final.fit(X_train_rf, y_train)

xgb_final = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05,
                        use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_final.fit(X_train, y_train)

svm_final = SVC(kernel='rbf', C=100, gamma=0.01, class_weight='balanced', probability=True, random_state=42)
svm_final.fit(X_train, y_train)

# Get test set base model predictions (probabilities)
print("\nPredicting on test set with base models...")
rf_test_probs = rf_final.predict_proba(X_test_rf)
xgb_test_probs = xgb_final.predict_proba(X_test)
svm_test_probs = svm_final.predict_proba(X_test)

# Stack test set probabilities for meta-model prediction
meta_X_test = np.hstack([rf_test_probs, xgb_test_probs, svm_test_probs])

print("\nMeta-model predicting final test labels...")
final_preds = meta_model.predict(meta_X_test)

# Evaluate final stacked model
acc = accuracy_score(y_test, final_preds)
print(f"\n🏆 Final stacked model accuracy: {acc:.4f}")
print(classification_report(y_test, final_preds, target_names=["Normal", "Insomnia"]))