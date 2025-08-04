import numpy as np
import joblib
import os
from collections import Counter
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.base import clone
from sklearn.pipeline import Pipeline

# === Load Data ===
data_path = "../results/features_bipolar_all_with_ids.npz"
assert os.path.exists(data_path), f"❌ File not found: {data_path}"
data = np.load(data_path)
X, y, groups = data["X"], data["y"], data["subject_ids"]
print("✅ Loaded data.")
print(f"Shape of X: {X.shape}, y distribution: {Counter(y)}")

# === Split Train/Test ===
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# === Load Trained Models ===
svm_model = joblib.load("../results/svm_bipolar_all_model.joblib")
rf_model = joblib.load("../results/rf_bipolar_all_model.joblib")
print("✅ Loaded saved SVM and RF models.")

# === Voting Classifier ===
voting_clf = VotingClassifier(
    estimators=[
        ("svm", svm_model),
        ("rf", rf_model)
    ],
    voting="soft",  # use soft voting for probability averaging
    n_jobs=-1
)

# === Fit Ensemble (only fits outer wrapper, individual models are already trained) ===
voting_clf.fit(X_train, y_train)
print("✅ VotingClassifier ensemble ready.")

# === Save Voting Model ===
joblib.dump(voting_clf, "../results/voting_clf_bipolar_all_model.joblib")

# === Evaluate Ensemble ===
y_pred = voting_clf.predict(X_test)
print("=== VotingClassifier Test Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# === Per-Channel Accuracy ===
print("\n📊 Per-Channel Accuracy Evaluation:")
n_channels = 3
features_per_channel = X.shape[1] // n_channels
channel_names = ["F4A1", "O2A1", "C4A1"]

for i in range(n_channels):
    ch_X_train = X_train[:, i * features_per_channel:(i + 1) * features_per_channel]
    ch_X_test = X_test[:, i * features_per_channel:(i + 1) * features_per_channel]

    # You can clone RF model if you want to use its exact hyperparams
    base_rf = rf_model.named_steps["rf"]
    rf_ch = RandomForestClassifier(
        class_weight='balanced',
        n_estimators=base_rf.n_estimators,
        max_depth=base_rf.max_depth,
        min_samples_split=base_rf.min_samples_split,
        min_samples_leaf=base_rf.min_samples_leaf,
        max_features=base_rf.max_features,
        bootstrap=base_rf.bootstrap,
        random_state=42
    )
    rf_ch.fit(ch_X_train, y_train)
    ch_y_pred = rf_ch.predict(ch_X_test)

    print(f"\n--- Channel {channel_names[i]} ---")
    print("Accuracy:", accuracy_score(y_test, ch_y_pred))
    print(classification_report(y_test, ch_y_pred))