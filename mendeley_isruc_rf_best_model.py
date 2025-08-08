# model after the second hypertuning
import joblib
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Load saved model
model_path = os.path.join("..", "results", "rf_rfe_tuned_model_better.joblib")
model = joblib.load(model_path)

# Load data (for inference)
data = np.load(os.path.join("..", "results", "combined_18fts.npz"))
X, y, groups = data["features"], data["labels"], data["subject_ids"]

# Use the same selected features
selected_idx = [2, 7, 17, 32, 37]
X = X[:, selected_idx]

# Predict
y_probs = model["model"].predict_proba(X)[:, 1]
y_pred = (y_probs >= model["threshold"]).astype(int)

# Evaluate
print(classification_report(y, y_pred))
accuracy = accuracy_score(y, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")