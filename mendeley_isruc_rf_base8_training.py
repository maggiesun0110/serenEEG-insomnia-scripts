import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, cross_val_score
from xgboost import XGBClassifier

# === Paths ===
results_path = os.path.join("..", "results")
mendeley_path = os.path.join(results_path, "mendeley_8fts.npz")
isruc_path = os.path.join(results_path, "isruc_8fts.npz")
combined_path = os.path.join(results_path, "combined_8fts.npz")

# === Load both datasets ===
m = np.load(mendeley_path)
i = np.load(isruc_path)

# === Combine ===
X_combined = np.vstack([m["features"], i["features"]])
y_combined = np.hstack([m["labels"], i["labels"]])
subject_ids_combined = np.hstack([m["subject_ids"], i["subject_ids"]])

# === Save combined ===
np.savez(combined_path, features=X_combined, labels=y_combined, subject_ids=subject_ids_combined)
print(f"Combined dataset saved: {combined_path}")
print("Combined shapes:", X_combined.shape, y_combined.shape)

# === Load combined ===
data = np.load(combined_path)
X, y, groups = data["features"], data["labels"], data["subject_ids"]

# --- Choose model: "rf" or "xgb" ---
model_choice = "xgb"

if model_choice == "rf":
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    gss = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    scores = cross_val_score(rf, X, y, groups=groups, cv=gss)
    print("RF CV scores:", scores)
    print("Mean RF accuracy:", scores.mean())
    
    rf.fit(X, y)
    joblib.dump(rf, os.path.join(results_path, "rf_combined_8fts.joblib"))
    
    # Plot feature importances
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_n = min(15, len(importances))

    plt.figure(figsize=(10, 5))
    plt.title("Top Feature Importances (Random Forest)")
    plt.bar(range(top_n), importances[indices[:top_n]], align="center")
    plt.xticks(range(top_n), [f"F{idx}" for idx in indices[:top_n]], rotation=45)
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, "feature_importance_rf.png"))
    plt.show()

elif model_choice == "xgb":
    print("Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    gss = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    scores = cross_val_score(xgb, X, y, groups=groups, cv=gss)
    print("XGB CV scores:", scores)
    print("Mean XGB accuracy:", scores.mean())
    
    xgb.fit(X, y)
    # Plot feature importances
    importances = xgb.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_n = min(15, len(importances))

    plt.figure(figsize=(10, 5))
    plt.title("Top Feature Importances (XGBoost)")
    plt.bar(range(top_n), importances[indices[:top_n]], align="center")
    plt.xticks(range(top_n), [f"F{idx}" for idx in indices[:top_n]], rotation=45)
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, "feature_importance_xgb.png"))
    plt.show()
    joblib.dump(xgb, os.path.join(results_path, "xgb_combined_8fts.joblib"))

else:
    print("Unknown model_choice! Choose 'rf' or 'xgb'.")