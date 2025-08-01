import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

# === Load Data ===
data = np.load("../results/features_a1_advanced_all_with_ids.npz")
X, y, subject_ids = data["X"], data["y"], data["subject_ids"]

# === Group-Aware Train/Test Split ===
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=subject_ids))
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# === Build Pipeline ===
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(probability=True, random_state=42))
])

# === Focused Hyperparameter Grid ===
param_grid = [
    {
        "svm__kernel": ["rbf"],
        "svm__C": [0.1, 1, 10],
        "svm__gamma": ["scale", "auto", 0.01],
        "svm__class_weight": [None, "balanced"],
        "svm__shrinking": [True, False]
    },
    {
        "svm__kernel": ["poly"],
        "svm__C": [1, 10],
        "svm__degree": [2, 3],
        "svm__gamma": ["scale", 0.01],
        "svm__coef0": [0.0, 0.5],
        "svm__class_weight": [None],
        "svm__shrinking": [True]
    }
]

# === Grid Search on Training Set ===
grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)
grid.fit(X_train, y_train)

# === Evaluation ===
print("\nBest hyperparameters:")
print(grid.best_params_)

y_pred = grid.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Test Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# === Save Model ===
joblib.dump(grid.best_estimator_, "../results/svm_deeper_tuned_model.joblib")
print("✅ Best tuned SVM model saved.")