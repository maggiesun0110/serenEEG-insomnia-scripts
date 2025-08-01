import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import joblib

# --- Paths to your feature files and model save location ---
sleepedf_features_path = "../results/features_with_ids.npz"
mendeley_features_path = "../results/mendeley_a1_features.npz"
model_save_path = "../results/rf_combined_model.joblib"

# --- Load features and labels ---
print("Loading Sleep-EDF features...")
sleepedf_data = np.load(sleepedf_features_path)
X_sleep = sleepedf_data['X']
y_sleep = sleepedf_data['y']

print("Loading Mendeley features...")
mendeley_data = np.load(mendeley_features_path)
X_mend = mendeley_data['X']
y_mend = mendeley_data['y']

print(f"Sleep-EDF samples: {X_sleep.shape[0]}, features: {X_sleep.shape[1]}")
print(f"Mendeley samples: {X_mend.shape[0]}, features: {X_mend.shape[1]}")

# --- Function to pad feature arrays to the same number of columns ---
def pad_features(X_source, target_dim):
    n_samples, n_feats = X_source.shape
    if n_feats >= target_dim:
        return X_source[:, :target_dim]
    else:
        padding = np.zeros((n_samples, target_dim - n_feats))
        return np.hstack([X_source, padding])

max_features = max(X_sleep.shape[1], X_mend.shape[1])

X_sleep_pad = pad_features(X_sleep, max_features)
X_mend_pad = pad_features(X_mend, max_features)

# --- Combine datasets ---
X_combined = np.vstack([X_sleep_pad, X_mend_pad])
y_combined = np.hstack([y_sleep, y_mend])

print(f"Combined dataset shape: {X_combined.shape}, Labels shape: {y_combined.shape}")

# --- Split combined data into train/test ---
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y_combined, test_size=0.2, stratify=y_combined, random_state=42)

print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# --- Train Random Forest classifier ---
print("Training Random Forest classifier...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# --- Evaluate ---
print("Evaluating on test set...")
y_pred = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Normal", "Insomnia"]))

# --- Save the trained model ---
joblib.dump(rf, model_save_path)
print(f"Model saved to {model_save_path}")