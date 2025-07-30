import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
import joblib

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

#convert to binary
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

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, digits=4, target_names=["Wake", "Non-Wake"]))

print("🎯 Accuracy:", accuracy_score(y_test, y_pred))

model_path = 'random_forest_insomnia_model.joblib'
joblib.dump(clf, model_path)
print(f"Model saved to {model_path}")