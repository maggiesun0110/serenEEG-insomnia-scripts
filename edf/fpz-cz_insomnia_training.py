import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt

data = np.load("../results/features.npz")
X = data["X"]
y = data["y"]

#split intro train/test
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, stratify=y, random_state=42)

#initizize rf model
clf = RandomForestClassifier(n_estimators=100, random_state = 42)
#training
clf.fit(X_train, y_train)

#predict
y_pred = clf.predict(X_test)

#test
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits = 4))

#fetuer importance
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,6))
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices])
plt.xlabel("Feature Index")
plt.ylabel("importance")
plt.tight_layout()
plt.savefig("../results/feature_importance_insomnia.png")
plt.show()

joblib.dump(clf, "../results/random_forest_model.pkl")