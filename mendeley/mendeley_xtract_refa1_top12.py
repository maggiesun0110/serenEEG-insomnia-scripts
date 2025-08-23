import numpy as np
import os
import mne
from scipy.signal import welch

def compute_mobility(data):
    diff1 = np.diff(data)
    var_zero = np.var(data)
    var_d1 = np.var(diff1)
    return np.sqrt(var_d1 / var_zero)

def compute_alpha_beta_ratio(data, sf):
    freqs, psd = welch(data, sf, nperseg=sf*2)
    alpha = np.sum(psd[(freqs >= 8) & (freqs < 13)])
    beta  = np.sum(psd[(freqs >= 13) & (freqs < 30)])
    return alpha / beta if beta > 0 else 0

def compute_mean(data):
    return np.mean(data)

def compute_std(data):
    return np.std(data)

def compute_complexity(data):
    diff1 = np.diff(data)
    diff2 = np.diff(diff1)
    var_zero = np.var(data)
    var_d1 = np.var(diff1)
    var_d2 = np.var(diff2)
    mobility = np.sqrt(var_d1 / var_zero)
    complexity = np.sqrt(var_d2 / var_d1) / mobility
    return complexity

# Config
BASE_PATH = os.path.join("..", "..", "..", "data", "mendeley")
CHANNELS = ["F4A1", "C4A1", "O2A1"]
SF_TARGET = 200
EPOCH_LEN = 30  # seconds

X, y, subject_ids = [], [], []

for label_folder, label in [("normal", 0), ("insomnia", 1)]:
    folder_path = os.path.join(BASE_PATH, label_folder)
    for file in os.listdir(folder_path):
        if not file.endswith(".edf"):
            continue
        file_path = os.path.join(folder_path, file)
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        raw.resample(SF_TARGET)
        raw.pick_channels(CHANNELS)

        data = raw.get_data()
        segment_len = SF_TARGET * EPOCH_LEN
        total_segments = data.shape[1] // segment_len

        for i in range(total_segments):
            feats = []

            # F4A1 - mobility + alpha/beta
            seg_f4 = data[0, i*segment_len:(i+1)*segment_len]
            feats.append(compute_mobility(seg_f4))
            feats.append(compute_alpha_beta_ratio(seg_f4, SF_TARGET))

            # C4A1 - mobility + alpha/beta
            seg_c4 = data[1, i*segment_len:(i+1)*segment_len]
            feats.append(compute_mobility(seg_c4))
            feats.append(compute_alpha_beta_ratio(seg_c4, SF_TARGET))

            # O2A1 - mobility + alpha/beta + mean + std + complexity
            seg_o2 = data[2, i*segment_len:(i+1)*segment_len]
            feats.append(compute_mobility(seg_o2))
            feats.append(compute_alpha_beta_ratio(seg_o2, SF_TARGET))
            feats.append(compute_mean(seg_o2))
            feats.append(compute_std(seg_o2))
            feats.append(compute_complexity(seg_o2))

            X.append(feats)
            y.append(label)
            subject_ids.append(file)  # use filename as subject id

X = np.array(X)
y = np.array(y)
subject_ids = np.array(subject_ids)

print("Shape:", X.shape, y.shape, "Unique subjects:", len(np.unique(subject_ids)))
np.savez("../../results/mendeley_12fts.npz", features=X, labels=y, subject_ids=subject_ids)