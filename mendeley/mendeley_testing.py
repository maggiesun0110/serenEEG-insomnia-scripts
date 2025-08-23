import joblib
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

# --- Paths ---
model_path = "../results/final_insomnia_rf_model.pkl"
test_data_path = "../results/testset_features.npz"

# --- Load Model ---
print("Loading trained model...")
model = joblib.load(model_path)

# --- Load Test Features ---
print("📂 Loading test data...")
data = np.load(test_data_path)
X_test = data["X"]
y_test = data["y"]

# --- Check Shape ---
print(f"🔍 Test set size: {X_test.shape[0]} samples, {X_test.shape[1]} features per sample")

# --- Inference ---
print("⚙️ Running predictions...")
y_pred = model.predict(X_test)

# --- Evaluation ---
print("\n📈 Evaluation on Mendeley test set:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Normal", "Insomnia"]))