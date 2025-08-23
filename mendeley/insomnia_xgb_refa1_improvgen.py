import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap

from xgboost import XGBClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import (
    GridSearchCV,
    GroupShuffleSplit,
    LeaveOneGroupOut,
    cross_val_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# === Load extracted features ===
data = np.load("../../results/mendeley_richfeatures.npz")
X = data['features']
y = data['labels']
groups = data['subject_ids']

print(f"Loaded: X = {X.shape}, y = {y.shape}, subjects = {len(np.unique(groups))}")

# === Step 0: Per-subject normalization ===
print("Normalizing features per subject...")
X_norm = np.zeros_like(X)
for sid in np.unique(groups):
    idx = groups == sid
    scaler = StandardScaler()
    X_norm[idx] = scaler.fit_transform(X[idx])

# === Step 1: GridSearchCV on RFE features after normalization ===
print("Running Grid Search with GroupShuffleSplit...")

# We'll do per-fold RFE inside cross-val later, but for GridSearch we pick features globally first for speed:
base_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
rfe_global = RFE(base_model, n_features_to_select=8)
rfe_global.fit(X_norm, y)
selected_features = np.where(rfe_global.support_)[0]
X_rfe_global = X_norm[:, selected_features]

print(f"Global selected features (for grid search): {selected_features}")

param_grid = {
    'subsample': [0.6, 0.7, 0.8],
    'scale_pos_weight': [0.9, 1.0, 1.1, 1.0165809033733562],
    'reg_lambda': [0.05, 0.1, 0.2],
    'reg_alpha': [5, 10, 15],
    'n_estimators': [550, 600, 650],
    'min_child_weight': [2, 3, 4],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.03, 0.05, 0.07],
    'gamma': [0.5, 1, 1.5],
    'colsample_bytree': [0.4, 0.5, 0.6]
}

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=gss,
    n_jobs=-1,
    verbose=2,
    refit=True
)

grid_search.fit(X_rfe_global, y, groups=groups)
print("✅ Best GridSearch Parameters:")
print(grid_search.best_params_)
print(f"Best CV Accuracy: {grid_search.best_score_ * 100:.2f}%")

# Save best model after grid search on global RFE
joblib.dump(grid_search.best_estimator_, "../../results/insomnia_xgb_improvgen.joblib")

# === Step 2: LOGO CV with per-fold normalization + per-fold RFE ===
print("Running LOGO cross-validation with per-fold normalization and RFE...")

logo = LeaveOneGroupOut()
accuracies = []
all_preds = []
all_trues = []

for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Normalize per fold
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)

    # Per-fold RFE
    base_model_fold = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, **grid_search.best_params_)
    rfe_fold = RFE(base_model_fold, n_features_to_select=8)
    X_train_rfe = rfe_fold.fit_transform(X_train_norm, y_train)
    X_test_rfe = rfe_fold.transform(X_test_norm)

    # Train model with best params on selected features
    model_fold = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, **grid_search.best_params_)
    model_fold.fit(X_train_rfe, y_train)

    preds = model_fold.predict(X_test_rfe)

    acc = accuracy_score(y_test, preds)
    accuracies.append(acc)
    all_preds.extend(preds)
    all_trues.extend(y_test)

    # Save each fold model
    joblib.dump(model_fold, f"../../results/improvgen_fold{fold}.joblib")

mean_acc = np.mean(accuracies)
std_acc = np.std(accuracies)
print(f"LOGO Accuracy: {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%")
np.savez("../../results/logo_scores_improvgen.npz", accuracy=mean_acc, std=std_acc, predictions=all_preds, truths=all_trues)

# === Step 3: SHAP on full data with global selected features ===
print("Computing SHAP values on globally selected features...")

# Use global selected features to retrain on whole normalized data for SHAP interpretability
model_shap = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, **grid_search.best_params_)
X_shap = X_norm[:, selected_features]
model_shap.fit(X_shap, y)

explainer = shap.Explainer(model_shap)
shap_values = explainer(X_shap)

plt.figure()
shap.summary_plot(shap_values.values, X_shap, show=False)
plt.tight_layout()
plt.savefig("../../results/shap_summary_improvgen.png")
print("SHAP summary plot saved.")

print("Saving per-subject SHAP plots...")
for sid in np.unique(groups):
    idx = groups == sid
    shap.summary_plot(shap_values.values[idx], X_shap[idx], show=False)
    plt.savefig(f"../../results/shap_subject_{sid}.png")
    plt.clf()

# === Step 4: Compare simpler models on globally selected features ===
print("Comparing simpler models...")
rf = RandomForestClassifier(random_state=42)
lr = LogisticRegression(max_iter=1000)
rf_scores = cross_val_score(rf, X_shap, y, groups=groups, cv=logo, scoring='accuracy')
lr_scores = cross_val_score(lr, X_shap, y, groups=groups, cv=logo, scoring='accuracy')
print(f"Random Forest LOGO: {rf_scores.mean() * 100:.2f}% ± {rf_scores.std() * 100:.2f}%")
print(f"Logistic Regression LOGO: {lr_scores.mean() * 100:.2f}% ± {lr_scores.std() * 100:.2f}%")

print("✅ All done.")