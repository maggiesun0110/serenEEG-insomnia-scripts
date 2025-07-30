import os
from scipy.io import loadmat
import numpy as np
import pandas as pd
from openpyxl import load_workbook
import matplotlib.pyplot as plt

# Pick one subject folder
mat_data = loadmat('../data/subject002/subject02.mat')
eeg = mat_data['F3_A2']
labels = np.loadtxt('../data/subject002/subject002_labels_1.txt')

# For EEG epochs (length = 934)
middle_start = (934 // 2) - (30 // 2)  # center - half window
middle_end = middle_start + 30

# Assuming clean_labels is labels trimmed to length 934
clean_labels = labels[:934]

print("Middle 30 EEG epochs (showing first 10 samples each):")
for i in range(middle_start, middle_end):
    print(f"Epoch {i}: {eeg[i][:10]}")

print("\nMiddle 30 labels:")
print(clean_labels[middle_start:middle_end])