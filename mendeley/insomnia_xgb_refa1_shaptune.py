import numpy as np
from xgboost import XGBClassifier
import shap

# Load data again
data = np.load("../../results/mendeley_12fts.npz")
X = data['features']
y = data['labels']

print(X.shape)

# RFE-selected 6 features
rfe_mask = np.array([True, True, True, False, False, True, False, True, True])
X_rfe = X[:, rfe_mask]

# Use your best params from GridSearch
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
    'colsample_bytree': 0.5,
    'use_label_encoder': False,
    'eval_metric': 'logloss',
    'random_state': 42
}

# Fit model directly
model = XGBClassifier(**best_params)
model.fit(X_rfe, y)

# SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_rfe)

# Feature names
feature_names = ["PSD_Alpha", "PSD_Beta", "Hjorth_Activity", "Spectral_Entropy", "Hjorth_Mobility", "Band_Ratio"]

# Plots
shap.summary_plot(shap_values, X_rfe, feature_names=feature_names)
shap.summary_plot(shap_values, X_rfe, feature_names=feature_names, plot_type="bar")