#best model rn 65.92
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
import matplotlib.pyplot as plt
import pandas as pd

# Load data
data = np.load("../../results/mendeley_12fts.npz")
X = data['features']
y = data['labels']
groups = data['subject_ids']

# RFE mask for 6 features (your best subset)
rfe_mask = np.array([True, True, True, False, False, True, False, True, True])
X_rfe = X[:, rfe_mask]

# Group-aware splitter
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# Best params from your randomized search
best_params = {
    'subsample': 0.7,
    'scale_pos_weight': 1.0165809033733562,
    'reg_lambda': 0.1,
    'reg_alpha': 10,
    'n_estimators': 600,
    'min_child_weight': 3,
    'max_depth': 4,
    'learning_rate': 0.05,
    'gamma': 1,
    'colsample_bytree': 0.5
}

# Define a narrow grid around those params
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

# Model setup
xgb_clf = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

# Setup GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb_clf,
    param_grid=param_grid,
    scoring='accuracy',
    cv=gss,
    verbose=2,
    n_jobs=-1,
    refit=True
)

# Run grid search
grid_search.fit(X_rfe, y, groups=groups)

print("Best parameters from GridSearch:")
print(grid_search.best_params_)
print(f"Best CV accuracy: {grid_search.best_score_ * 100:.2f}%")

# Plot results
cv_results = pd.DataFrame(grid_search.cv_results_)

plt.figure(figsize=(12, 6))
plt.plot(cv_results['rank_test_score'], cv_results['mean_test_score'], 'o-', label='Mean Test Accuracy')
plt.gca().invert_xaxis()
plt.xlabel('Rank (Best=1)')
plt.ylabel('Mean CV Accuracy')
plt.title('GridSearchCV Accuracy Rankings')
plt.grid(True)
plt.show()

import joblib

joblib.dump(xgb_clf, '../../results/insomnia_xgb_model_rfe6.pkl')
print("Model saved to ../../results/insomnia_xgb_model_rfe6.pkl")

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import cross_val_score

logo = LeaveOneGroupOut()

scores = cross_val_score(
    grid_search.best_estimator_,  # or any model
    X_rfe, y,
    groups=groups,
    cv=logo,
    scoring='accuracy'
)

print(f"LOGO Accuracy: {scores.mean() * 100:.2f}% ± {scores.std() * 100:.2f}%")

#randomizer that got 65.04
# import numpy as np
# from xgboost import XGBClassifier
# from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV
# import matplotlib.pyplot as plt
# import pandas as pd

# # Load data
# data = np.load("../../results/mendeley_12fts.npz")
# X = data['features']
# y = data['labels']
# groups = data['subject_ids']

# # RFE mask for 6 features (your best subset)
# rfe_mask = np.array([True, True, True, False, False, True, False, True, True])
# X_rfe = X[:, rfe_mask]

# # Group-aware splitter
# gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# # Expanded parameter distributions
# param_dist = {
#     'n_estimators': [100, 200, 300, 400, 500, 600, 700],
#     'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10],
#     'learning_rate': [0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2],
#     'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
#     'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
#     'reg_alpha': [0, 0.01, 0.1, 0.5, 1, 5, 10],
#     'reg_lambda': [0.1, 0.5, 1, 3, 5, 7, 10],
#     'min_child_weight': [1, 3, 5, 7, 10, 15],
#     'gamma': [0, 0.1, 0.3, 0.5, 1, 5],
#     'scale_pos_weight': [(y == 0).sum() / (y == 1).sum()]
# }

# # Create the model
# xgb_clf = XGBClassifier(
#     use_label_encoder=False,
#     eval_metric='logloss',
#     random_state=42
# )

# # Setup RandomizedSearchCV with more iterations for bigger space
# random_search = RandomizedSearchCV(
#     estimator=xgb_clf,
#     param_distributions=param_dist,
#     n_iter=100,             # Increased from 50 to 100 iterations
#     scoring='accuracy',
#     cv=gss,
#     verbose=2,
#     random_state=42,
#     n_jobs=-1,
#     refit=True
# )

# # Run the search
# random_search.fit(X_rfe, y, groups=groups)

# print("Best parameters from expanded RandomizedSearch:")
# print(random_search.best_params_)
# print(f"Best CV accuracy: {random_search.best_score_ * 100:.2f}%")

# # Plotting the results
# cv_results = pd.DataFrame(random_search.cv_results_)

# plt.figure(figsize=(12, 6))
# plt.plot(cv_results['rank_test_score'], cv_results['mean_test_score'], 'o-', label='Mean Test Accuracy')
# plt.gca().invert_xaxis()
# plt.xlabel('Rank (Best=1)')
# plt.ylabel('Mean CV Accuracy')
# plt.title('Expanded RandomizedSearchCV Accuracy Rankings')
# plt.grid(True)
# plt.show()