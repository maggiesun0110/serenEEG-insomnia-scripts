import numpy as np
import os
import mne
from scipy.signal import welch
from scipy.integrate import simpson

# === Feature functions ===

def bandpower(data, sf, bands):
    freqs, psd = welch(data, sf, nperseg=sf * 2)
    return [
        simpson(psd[(freqs >= low) & (freqs <= high)],
                freqs[(freqs >= low) & (freqs <= high)])
        for low, high in bands
    ]

def hjorth_parameters(data):
    diff1 = np.diff(data)
    diff2 = np.diff(diff1)
    var_zero = np.var(data)
    var_d1 = np.var(diff1)
    var_d2 = np.var(diff2)
    mobility = np.sqrt(var_d1 / var_zero)
    complexity = np.sqrt(var_d2 / var_d1) / (mobility + 1e-6)
    return [var_zero, mobility, complexity]

# === Config ===
BASE_PATH = os.path.join("..", "..", "..", "data", "mendeley")
CHANNELS = ["F4A1", "C4A1", "O2A1"]
SF_TARGET = 200
EPOCH_LEN = 30  # seconds
BANDS = [(0.5, 4), (4, 8), (8, 12), (12, 30), (30, 50)]
RESULTS_PATH = os.path.join("..", "..", "results")
OUTFILE = os.path.join(RESULTS_PATH, "mendeley_8fts.npz")

# === Storage ===
X, y, subject_ids = [], [], []

# === Main extraction loop ===
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

        subject_id = os.path.splitext(file)[0] 

        for i in range(total_segments):
            feats = []
            for ch_idx in range(len(CHANNELS)):
                seg = data[ch_idx, i * segment_len:(i + 1) * segment_len]
                feats.extend(bandpower(seg, SF_TARGET, BANDS))
                feats.extend(hjorth_parameters(seg))
            X.append(feats)
            y.append(label)
            subject_ids.append(subject_id)

# === Save ===
os.makedirs(RESULTS_PATH, exist_ok=True)
np.savez(OUTFILE,
         features=np.array(X),
         labels=np.array(y),
         subject_ids=np.array(subject_ids))

print(f"Saved to: {OUTFILE}")
print(f"Shape: features = {np.array(X).shape}, labels = {np.array(y).shape}, subjects = {len(set(subject_ids))} unique")