import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, cross_val_score
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier

# === Paths ===
results_path = os.path.join("..", "results")
mendeley_path = os.path.join(results_path, "mendeley_18fts.npz")
isruc_path = os.path.join(results_path, "isruc_18fts.npz")
combined_path = os.path.join(results_path, "combined_18fts.npz")

# === Load datasets ===
m = np.load(mendeley_path)
i = np.load(isruc_path)

X_combined = np.vstack([m["features"], i["features"]])
y_combined = np.hstack([m["labels"], i["labels"]])
subject_ids_combined = np.hstack([m["subject_ids"], i["subject_ids"]])

# === Save combined ===
np.savez(combined_path, features=X_combined, labels=y_combined, subject_ids=subject_ids_combined)
print("Combined dataset saved:", combined_path)
print("Combined shapes:", X_combined.shape, y_combined.shape)

# === Load combined ===
data = np.load(combined_path)
X, y, groups = data["features"], data["labels"], data["subject_ids"]

# === Feature Names ===
channels = ["F4A1", "C4A1", "O2A1"]
base_feats = [
    "delta", "theta", "alpha", "beta", "gamma",                  # 5 PSD
    "δ/θ", "δ/α", "α/β", "θ/α",                                   # 4 Ratios
    "activity", "mobility", "complexity",                        # 3 Hjorth
    "mean", "std", "skew"                                        # 3 Stats
]
feature_names = [f"{ch}_{f}" for ch in channels for f in base_feats]

def run_model_with_auto_rfe(name, model, X, y, groups):
    print(f"\n--- Running {name.upper()} with Automated RFE Feature Selection ---")
    max_features = X.shape[1]
    gss = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

    best_score = 0
    best_num_features = None
    best_rfe = None

    for n_feat in range(1, max_features + 1):
        rfe = RFE(estimator=model, n_features_to_select=n_feat, step=1)
        X_selected = rfe.fit_transform(X, y)
        scores = cross_val_score(model, X_selected, y, groups=groups, cv=gss)
        mean_score = scores.mean()
        print(f"Features: {n_feat:2d} - CV Mean Accuracy: {mean_score:.4f}")

        if mean_score > best_score:
            best_score = mean_score
            best_num_features = n_feat
            best_rfe = rfe

    print(f"\nBest number of features: {best_num_features} with CV accuracy: {best_score:.4f}")

    # Use best RFE to get final selected features and retrain
    support_idx = np.where(best_rfe.support_)[0]
    print("Selected feature indices:", support_idx)
    print("Selected feature names:")
    for idx in support_idx:
        print(f"  {idx}: {feature_names[idx]}")

    X_best = best_rfe.transform(X)
    model.fit(X_best, y)
    joblib.dump(model, os.path.join(results_path, f"{name}_auto_rfe_model.joblib"))

    # Plot feature importances
    importances = model.feature_importances_
    top_n = min(15, len(importances))
    sorted_idx = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(10, 5))
    plt.title(f"Top Feature Importances ({name.upper()} + Auto RFE)")
    plt.bar(range(top_n), importances[sorted_idx], align="center")
    plt.xticks(range(top_n), [feature_names[support_idx[i]] for i in sorted_idx], rotation=45)
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f"feature_importance_{name}_auto_rfe.png"))
    plt.show()

# === Run models ===
run_model_with_auto_rfe("rf", RandomForestClassifier(n_estimators=100, random_state=42), X, y, groups)
run_model_with_auto_rfe("xgb", XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
), X, y, groups)