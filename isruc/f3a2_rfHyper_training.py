import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay 

# Adjust if needed
results_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'results', 'batches'))

# Load all batches
feature_batches = []
label_batches = []

for i in range(1, 5):
    f = np.load(os.path.join(results_path, f'features_batch_{i}.npy'))
    l = np.load(os.path.join(results_path, f'labels_batch_{i}.npy'))
    feature_batches.append(f)
    label_batches.append(l)

# Combine
all_features = np.vstack(feature_batches)
all_labels = np.hstack(label_batches)

# Convert to binary
all_labels = np.where(all_labels == 0.0, 0, 1)  # 0 = Wake, 1 = Non-Wake

print("Combined features shape:", all_features.shape)
print("Combined labels shape:", all_labels.shape)

scaler = StandardScaler()
features_scaled = scaler.fit_transform(all_features)

print("Scaled feature shape:", features_scaled.shape)

np.save(os.path.join(results_path, 'features_all_scaled.npy'), features_scaled)
np.save(os.path.join(results_path, 'labels_all.npy'), all_labels)

#--------training

X_train, X_test, y_train, y_test = train_test_split(
    features_scaled, all_labels, test_size=0.2, random_state=42, stratify=all_labels
)

# ======= INSERT GRIDSEARCHCV HYPERPARAMETER TUNING HERE =======
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [150, 175, 200, 225],
    'max_depth': [18, 20, 22],
    'min_samples_split': [2, 3],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

clf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(
    estimator=clf,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2,
    scoring='accuracy'
)

grid_search.fit(X_train, y_train)

print("Best hyperparameters:", grid_search.best_params_)
print("Best CV accuracy:", grid_search.best_score_)

best_clf = grid_search.best_estimator_
y_pred = best_clf.predict(X_test)

print("\n📊 Classification Report (Best Model):")
print(classification_report(y_test, y_pred, digits=4, target_names=["Wake", "Non-Wake"]))

print("🎯 Test Accuracy:", accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Wake", "Non-Wake"])
disp.plot(cmap="Blues", values_format='d')  # 'd' = integer formatting

import matplotlib.pyplot as plt
plt.title("Confusion Matrix - Best Random Forest Model")
plt.tight_layout()
plt.show()

model_path = 'random_forest_insomnia_model_best.joblib'
joblib.dump(best_clf, model_path)
print(f"Best model saved to {model_path}")