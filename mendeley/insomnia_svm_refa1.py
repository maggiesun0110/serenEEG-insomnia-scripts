#need to run this
import numpy as np
import joblib
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter

# === Load Data ===
data = np.load("../results/features_bipolar_all_with_ids.npz")
X, y, subject_ids = data["X"], data["y"], data["subject_ids"]
print("✅ Loaded data.")
print(f"Shape of X: {X.shape}, y distribution: {Counter(y)}")

# === Train/Test Split ===
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, subject_ids))
X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]
groups_train = subject_ids[train_idx]

# === Pipeline ===
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("select", SelectKBest(score_func=f_classif)),
    ("svm", SVC(class_weight="balanced", probability=True))
])

# === Grid Search Parameters ===
param_grid = {
    "select__k": [10, 20, 30, 40, 50, "all"],
    "svm__kernel": ["rbf"],  # try "poly" if you want to explore
    "svm__C": [0.01, 0.1, 1, 10, 100, 1000],
    "svm__gamma": [1e-4, 1e-3, 1e-2, 0.1, 1]
}

# === Cross-Validation ===
cv = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

gs = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="accuracy",  # optionally try "f1", "balanced_accuracy"
    cv=cv.split(X_train, y_train, groups=groups_train),
    verbose=1,
    n_jobs=-1
)

# === Train ===
gs.fit(X_train, y_train)

# === Save Best Model ===
joblib.dump(gs.best_estimator_, "../results/svm_bipolar_all_model_v3.joblib")
print("✅ Model saved.")
print("✅ Best Params:", gs.best_params_)
print("✅ Best CV Accuracy:", gs.best_score_)

# === Evaluate ===
y_pred = gs.predict(X_test)
print("=== Test Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))