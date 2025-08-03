import matplotlib.pyplot as plt
import numpy as np

# Data for the two models to compare
models = {
    "Ensembled Model": {
        "accuracy": 0.5979,
        "metrics": {
            "Normal": {"precision": 0.62, "recall": 0.86, "f1": 0.72},
            "Insomnia": {"precision": 0.47, "recall": 0.20, "f1": 0.28},
        }
    },
    "Optimized SVM": {
        "accuracy": 0.7004,
        "metrics": {
            "Normal": {"precision": 0.71, "recall": 0.87, "f1": 0.78},
            "Insomnia": {"precision": 0.69, "recall": 0.44, "f1": 0.53},
        },
        "hyperparams": {
            'C': 0.1,
            'class_weight': 'balanced',
            'gamma': 0.01,
            'kernel': 'rbf',
            'shrinking': True
        }
    }
}

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Model Comparison: Ensembled vs Optimized SVM', fontsize=16)

# Colors for the bars
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# 1. Metrics comparison plot
for i, (model_name, model_data) in enumerate(models.items()):
    # Extract metrics
    classes = list(model_data["metrics"].keys())
    metrics = ['precision', 'recall', 'f1']
    values = {
        'precision': [model_data["metrics"][c]["precision"] for c in classes],
        'recall': [model_data["metrics"][c]["recall"] for c in classes],
        'f1': [model_data["metrics"][c]["f1"] for c in classes]
    }
    
    # Set bar width and positions
    bar_width = 0.35
    index = np.arange(len(classes))
    
    # Plot bars for each metric
    for j, metric in enumerate(metrics):
        offset = (j - 1) * bar_width
        ax1.bar(index + offset, values[metric], bar_width, 
                label=f'{metric.capitalize()} ({model_name})' if i == 0 else '',
                color=colors[j], alpha=0.7 if i == 1 else 1)
    
    # Add accuracy as text
    ax1.text(i * 0.5 + 0.25, 1.05, f'{model_name} Accuracy: {model_data["accuracy"]:.4f}', 
             ha='center', va='bottom', transform=ax1.transAxes, fontsize=12)

# Customize the metrics plot
ax1.set_title('Performance Metrics Comparison', fontsize=14)
ax1.set_xticks(index)
ax1.set_xticklabels(classes)
ax1.set_ylim(0, 1.1)
ax1.set_ylabel('Score')
ax1.legend(loc='lower right')
ax1.grid(True, axis='y', linestyle='--', alpha=0.7)

# Add value labels (optional)
for bars in ax1.containers:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9)

# 2. Accuracy comparison with hyperparameters
accuracies = [models[model]["accuracy"] for model in models]
model_names = list(models.keys())

# Create the bar plot
bars = ax2.bar(model_names, accuracies, color=['#4c72b0', '#55a868'])

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom', fontsize=12)

# Add hyperparameters information for the SVM
hyperparams = models["Optimized SVM"]["hyperparams"]
hyperparam_text = "Optimized SVM Hyperparameters:\n" + "\n".join([f"{k}: {v}" for k, v in hyperparams.items()])
ax2.text(0.5, 0.3, hyperparam_text, 
         ha='center', va='center', transform=ax2.transAxes,
         bbox=dict(facecolor='white', alpha=0.8), fontsize=10)

# Customize the accuracy plot
ax2.set_title('Accuracy Comparison with Hyperparameters', fontsize=14)
ax2.set_ylabel('Accuracy')
ax2.set_ylim(0.5, 0.75)
ax2.grid(True, axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()