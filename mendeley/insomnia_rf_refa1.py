import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter

# === Load Data ===
data_path = "../results/features_bipolar_all_with_ids.npz"
assert os.path.exists(data_path), f"❌ File not found: {data_path}"

data = np.load(data_path)
X = data["X"]         # shape: (num_samples, 54)
y = data["y"]         # shape: (num_samples,)
groups = data["subject_ids"]

print("✅ Loaded data.")
print(f"Shape of X: {X.shape}, y distribution: {Counter(y)}")

# === Create RF Pipeline ===
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Optional for RF, but included for consistency
    ('rf', RandomForestClassifier(class_weight='balanced', random_state=42))
])

# === Grid Search Parameters ===
param_grid = {
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [None, 10, 20],
    'rf__min_samples_split': [2, 5],
    'rf__min_samples_leaf': [1, 2],
}

cv = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

grid = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)

grid.fit(X, y, groups=groups)

# === Save Best Model ===
model_path = "../results/rf_bipolar_all_model.joblib"
joblib.dump(grid.best_estimator_, model_path)

print("✅ Model saved to:", model_path)
print("✅ Best Params:", grid.best_params_)
print("✅ Best CV Accuracy:", grid.best_score_)

# === Evaluate on Final Split ===
train_idx, test_idx = next(cv.split(X, y, groups=groups))
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

y_pred = grid.best_estimator_.predict(X_test)
print("=== Test Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))