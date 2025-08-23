import numpy as np

# Load the file
data = np.load("/Users/maggiesun/Downloads/research/SerenEEG/insomnia_rf/results/features.npz")

# Check what keys are inside
print("Keys:", data.files)

# Load X and y
X = data["X"]
y = data["y"]

# Show shapes
print("X shape:", X.shape)  #(num_samples, num_features)
print("y shape:", y.shape)  #(num_samples,)

print("labels 140000 - 140200:", y[140000:140200])

unique, counts = np.unique(y, return_counts=True)
print("Label distribution:")
for label, count in zip(unique, counts):
    print(f"Label {int(label)}: {count} samples")

# Print the first sample and its label
print("Sample features (first one):")
print(X[0])

print("Label (first one):")
print(y[0])