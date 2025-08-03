import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
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

# === Build Pipeline with placeholder for feature selector ===
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("feature_sel", SelectKBest()),  # placeholder, will be overridden by GridSearch
    ("svm", SVC(probability=True, random_state=42))
])

# === Hyperparameter Grid ===
param_grid = [
    # SelectKBest option
    {
        "feature_sel": [SelectKBest(score_func=f_classif)],
        "feature_sel__k": [5, 10, 15, "all"],

        "svm__kernel": ["rbf"],
        "svm__C": [0.1, 1, 10],
        "svm__gamma": ["scale", "auto", 0.01],
        "svm__class_weight": [None, "balanced"],
        "svm__shrinking": [True, False]
    },
    # PCA option
    {
        "feature_sel": [PCA()],
        "feature_sel__n_components": [5, 10, 15, 0.95],  # 0.95 means enough PCs to explain 95% variance

        "svm__kernel": ["rbf"],
        "svm__C": [0.1, 1, 10],
        "svm__gamma": ["scale", "auto", 0.01],
        "svm__class_weight": [None, "balanced"],
        "svm__shrinking": [True, False]
    },
    # Poly kernel without feature selection for comparison
    {
        "feature_sel": ["passthrough"],  # no feature selection
        "svm__kernel": ["poly"],
        "svm__C": [1, 10],
        "svm__degree": [2, 3],
        "svm__gamma": ["scale", 0.01],
        "svm__coef0": [0.0, 0.5],
        "svm__class_weight": [None],
        "svm__shrinking": [True]
    }
]

# === Grid Search on Training Set with more folds ===
grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,  # increased folds
    scoring="accuracy",
    n_jobs=-1,
    verbose=2
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