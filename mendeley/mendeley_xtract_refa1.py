from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import randint
import numpy as np
import joblib
from collections import Counter

# === Load Data ===
data = np.load("../results/features_bipolar_all_with_ids.npz")
X, y, groups = data["X"], data["y"], data["subject_ids"]
print("✅ Loaded data.")
print(f"Shape of X: {X.shape}, y distribution: {Counter(y)}")

# === Pipeline ===
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('select', SelectKBest(score_func=f_classif, k=40)),  # Tune k
    ('rf', RandomForestClassifier(class_weight='balanced', random_state=42))
])

# === Parameter Distribution ===
param_dist = {
    'rf__n_estimators': randint(200, 600),
    'rf__max_depth': [5, 10, 15],
    'rf__min_samples_split': [2, 4, 6],
    'rf__min_samples_leaf': [1, 2, 3],
    'rf__max_features': ['sqrt', 'log2'],
    'rf__bootstrap': [False]  # <– help generalize with high variance data
}

# === Cross-validation ===
cv = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=40,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2,
    random_state=42
)

search.fit(X, y, groups=groups)

# === Save + Evaluate ===
joblib.dump(search.best_estimator_, "../results/rf_bipolar_best_model.joblib")
print("✅ Model saved.")
print("✅ Best Params:", search.best_params_)
print("✅ Best CV Accuracy:", search.best_score_)

# Final holdout eval
train_idx, test_idx = next(cv.split(X, y, groups))
X_test, y_test = X[test_idx], y[test_idx]
y_pred = search.best_estimator_.predict(X_test)
print("=== Test Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))