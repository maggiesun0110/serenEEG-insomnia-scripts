import numpy as np
import os
import mne
from scipy.signal import welch
from scipy.integrate import simpson
from scipy.stats import entropy as shannon_entropy, skew, kurtosis

# === Feature functions ===
def bandpower(data, sf, band):
    freqs, psd = welch(data, sf, nperseg=sf*2)
    low, high = band
    mask = (freqs >= low) & (freqs <= high)
    return simpson(psd[mask], freqs[mask])

def full_features(signal, sf):
    freqs, psd = welch(signal, sf, nperseg=sf*2)
    psd_norm = psd / np.sum(psd)
    se = shannon_entropy(psd_norm)

    delta = bandpower(signal, sf, (0.5, 4))
    theta = bandpower(signal, sf, (4, 8))
    alpha = bandpower(signal, sf, (8, 12))
    beta = bandpower(signal, sf, (12, 30))

    ab_ratio = alpha / beta if beta > 0 else 0
    da_ratio = delta / alpha if alpha > 0 else 0

    activity = np.var(signal)
    diff1 = np.diff(signal)
    diff2 = np.diff(diff1)
    var_d1 = np.var(diff1)
    var_d2 = np.var(diff2)
    mobility = np.sqrt(var_d1 / activity) if activity > 0 else 0
    complexity = (np.sqrt(var_d2 / var_d1) / mobility) if var_d1 > 0 and mobility > 0 else 0

    sk = skew(signal)
    kurt = kurtosis(signal)

    return [
        delta, theta, alpha, beta,
        ab_ratio, da_ratio,
        activity, mobility, complexity,
        se, sk, kurt
    ]

# === Config ===
BASE_PATH = os.path.join("..", "..", "..", "data", "mendeley")
CHANNELS = ["F4A1", "C4A1", "O2A1"]
SF_TARGET = 200
EPOCH_LEN = 30  # seconds
segment_len = SF_TARGET * EPOCH_LEN

X, y, subject_ids = [], [], []
subject_counter = 0

# === Loop through all subjects ===
for label_folder, label in [("normal", 0), ("insomnia", 1)]:
    folder_path = os.path.join(BASE_PATH, label_folder)
    for filename in os.listdir(folder_path):
        if not filename.endswith(".edf"):
            continue

        raw = mne.io.read_raw_edf(os.path.join(folder_path, filename), preload=True, verbose=False)
        raw.resample(SF_TARGET)
        raw.pick_channels(CHANNELS)
        data = raw.get_data()  # shape: (3, timepoints)

        total_segments = data.shape[1] // segment_len

        for i in range(total_segments):
            feats = []
            for ch in range(3):
                seg = data[ch, i * segment_len:(i + 1) * segment_len]
                feats.extend(full_features(seg, SF_TARGET))

            X.append(feats)
            y.append(label)
            subject_ids.append(subject_counter)

        subject_counter += 1

# === Save ===
X = np.array(X)
y = np.array(y)
subject_ids = np.array(subject_ids)

print("Shape: features =", X.shape, ", labels =", y.shape, ", subjects =", len(np.unique(subject_ids)))
np.savez("../../results/mendeley_richfeatures.npz", features=X, labels=y, subject_ids=subject_ids)