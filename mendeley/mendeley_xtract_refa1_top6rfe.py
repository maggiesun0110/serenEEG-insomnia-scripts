import numpy as np
import os
import mne
from scipy.signal import welch
from scipy.integrate import simpson

def bandpower(data, sf, band):
    freqs, psd = welch(data, sf, nperseg=sf*2)
    low, high = band
    mask = (freqs >= low) & (freqs <= high)
    return simpson(psd[mask], freqs[mask])

def alpha_beta_ratio(data, sf):
    alpha = bandpower(data, sf, (8, 12))
    beta = bandpower(data, sf, (12, 30))
    return alpha / beta if beta > 0 else 0

def hjorth_parameters(data):
    diff1 = np.diff(data)
    diff2 = np.diff(diff1)
    var_zero = np.var(data)
    var_d1 = np.var(diff1)
    var_d2 = np.var(diff2)
    mobility = np.sqrt(var_d1 / var_zero) if var_zero > 0 else 0
    complexity = (np.sqrt(var_d2 / var_d1) / mobility) if var_d1 > 0 and mobility > 0 else 0
    return mobility, complexity

# === Config ===
BASE_PATH = os.path.join("..", "..", "..", "data", "mendeley")
CHANNELS = ["F4A1", "C4A1", "O2A1"]
SF_TARGET = 200
EPOCH_LEN = 30  # seconds
segment_len = SF_TARGET * EPOCH_LEN

X, y, subject_ids = [], [], []
subject_counter = 0

for label_folder, label in [("normal", 0), ("insomnia", 1)]:
    folder_path = os.path.join(BASE_PATH, label_folder)
    for filename in os.listdir(folder_path):
        if not filename.endswith(".edf"):
            continue

        raw = mne.io.read_raw_edf(os.path.join(folder_path, filename), preload=True, verbose=False)
        raw.resample(SF_TARGET)
        raw.pick_channels(CHANNELS)
        data = raw.get_data()

        total_segments = data.shape[1] // segment_len

        for i in range(total_segments):
            feats = []

            # F4A1
            seg_f4 = data[0, i * segment_len:(i + 1) * segment_len]
            mob_f4, _ = hjorth_parameters(seg_f4)
            ab_f4 = alpha_beta_ratio(seg_f4, SF_TARGET)

            # C4A1
            seg_c4 = data[1, i * segment_len:(i + 1) * segment_len]
            mob_c4, _ = hjorth_parameters(seg_c4)

            # O2A1
            seg_o2 = data[2, i * segment_len:(i + 1) * segment_len]
            ab_o2 = alpha_beta_ratio(seg_o2, SF_TARGET)
            std_o2 = np.std(seg_o2)
            _, comp_o2 = hjorth_parameters(seg_o2)

            feats.extend([mob_f4, ab_f4, mob_c4, ab_o2, std_o2, comp_o2])
            X.append(feats)
            y.append(label)
            subject_ids.append(subject_counter)

        subject_counter += 1

# === Save ===
X = np.array(X)
y = np.array(y)
subject_ids = np.array(subject_ids)

print("Shape: features =", X.shape, ", labels =", y.shape, ", subjects =", len(np.unique(subject_ids)))
np.savez("../../results/mendeley_rfe6.npz", features=X, labels=y, subject_ids=subject_ids)