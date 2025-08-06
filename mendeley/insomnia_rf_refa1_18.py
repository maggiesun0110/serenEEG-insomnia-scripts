import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

# === Load data ===
data = np.load("../../results/mendeley_18fts.npz")
X = data['features']
y = data['labels']
groups = data['subject_ids']

# === Group-aware split ===
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# === Pipeline: Feature Selection + Random Forest ===
pipe = Pipeline([
    ('select', SelectKBest(score_func=f_classif, k=10)),  # 👈 top 10 features
    ('clf', RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        class_weight='balanced'
    ))
])

pipe.fit(X_train, y_train)

# === Evaluate ===
y_pred = pipe.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))

import matplotlib.pyplot as plt
import numpy as np

# === Step 1: Define full feature names ===
channels = ["F4A1", "C4A1", "O2A1"]
feature_types = [
    "delta", "theta", "alpha", "beta", "gamma",
    "delta/theta", "delta/alpha", "alpha/beta", "theta/alpha",
    "activity", "mobility", "complexity",
    "mean", "std", "skew"
]
all_feature_names = [f"{ch}_{ft}" for ch in channels for ft in feature_types]

# === Step 2: Get selected mask and scores ===
selector = pipe.named_steps['select']
mask = selector.get_support()           # boolean mask
scores = selector.scores_              # f_classif scores
selected_names = np.array(all_feature_names)[mask]
selected_scores = scores[mask]

# === Step 3: Sort by score and plot ===
sorted_idx = np.argsort(selected_scores)[::-1]
sorted_names = selected_names[sorted_idx]
sorted_scores = selected_scores[sorted_idx]

plt.figure(figsize=(10, 5))
plt.barh(sorted_names, sorted_scores, color='navy')
plt.xlabel("ANOVA F-score")
plt.title("Top Selected EEG Features by SelectKBest (f_classif)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()