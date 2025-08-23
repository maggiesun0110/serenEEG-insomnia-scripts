import numpy as np
import matplotlib.pyplot as plt

# Load LOSO results
results = np.load("../results/loso_results.npz", allow_pickle=True)
accuracies = results['accuracies']
subjects = results['test_subjects']

# Sort by subject ID (optional, for prettier plots)
sorted_indices = np.argsort(subjects)
subjects_sorted = subjects[sorted_indices]
accuracies_sorted = accuracies[sorted_indices]

# Plot
plt.figure(figsize=(12, 6))
plt.bar(range(len(accuracies_sorted)), accuracies_sorted, color='skyblue')
plt.xticks(range(len(subjects_sorted)), subjects_sorted, rotation=45)
plt.ylim(0, 1.05)
plt.ylabel("Accuracy")
plt.xlabel("Test Subject ID")
plt.title("LOSO Accuracy per Subject")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()