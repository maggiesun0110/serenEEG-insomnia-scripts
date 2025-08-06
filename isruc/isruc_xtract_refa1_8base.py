import os
import numpy as np
from scipy.io import loadmat
from scipy.signal import welch, butter, filtfilt, resample
from scipy.integrate import simpson

# === Feature Functions ===
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
    mobility = np.sqrt(var_d1 / var_zero) if var_zero > 0 else 0
    complexity = np.sqrt(var_d2 / var_d1) / (mobility + 1e-6) if var_d1 > 0 else 0
    return [var_zero, mobility, complexity]

def bandpass_filter(data, sf, low=0.5, high=40):
    b, a = butter(N=4, Wn=[low, high], btype='band', fs=sf)
    return filtfilt(b, a, data)

def preprocess(data, orig_sf, target_sf=200):
    data = bandpass_filter(data, orig_sf)
    if orig_sf != target_sf:
        duration = len(data) / orig_sf
        data = resample(data, int(duration * target_sf))
    return data

# === Config ===
BASE_PATH = os.path.join("..", "..", "..", "data", "ISRUC sleep", "data")
RESULTS_PATH = os.path.join("..", "..", "results")
CHANNELS = ["F4_A1", "C4_A1", "O2_A1"]
EPOCH_LEN = 30  # seconds
SF_TARGET = 200
EPOCH_SAMPLES = EPOCH_LEN * SF_TARGET
BANDS = [(0.5, 4), (4, 8), (8, 12), (12, 30), (30, 50)]
OUTFILE = os.path.join(RESULTS_PATH, "isruc_8fts.npz")

X, y, subject_ids = [], [], []

# === Main Loop ===
for subj in sorted(os.listdir(BASE_PATH)):
    subj_path = os.path.join(BASE_PATH, subj)
    if not os.path.isdir(subj_path):
        continue

    print("Processing:", subj)
    mat_file = next((f for f in os.listdir(subj_path) if f.endswith(".mat")), None)
    if not mat_file:
        print("No .mat file found.")
        continue

    mat_path = os.path.join(subj_path, mat_file)
    mat = loadmat(mat_path)

    try:
        raw_f4 = mat["F4_A1"].flatten()
        raw_c4 = mat["C4_A1"].flatten()
        raw_o2 = mat["O2_A1"].flatten()
    except KeyError:
        print(f"Missing channel in {subj}")
        continue

    orig_sf = mat.get("sampling_frequency", SF_TARGET)
    if isinstance(orig_sf, np.ndarray):
        orig_sf = int(orig_sf.flatten()[0])

    f4 = preprocess(raw_f4, orig_sf, SF_TARGET)
    c4 = preprocess(raw_c4, orig_sf, SF_TARGET)
    o2 = preprocess(raw_o2, orig_sf, SF_TARGET)

    min_len = min(len(f4), len(c4), len(o2))
    n_epochs = min_len // EPOCH_SAMPLES
    if n_epochs == 0:
        print("Too short for 30s epochs.")
        continue

    f4_epochs = f4[:n_epochs * EPOCH_SAMPLES].reshape(n_epochs, EPOCH_SAMPLES)
    c4_epochs = c4[:n_epochs * EPOCH_SAMPLES].reshape(n_epochs, EPOCH_SAMPLES)
    o2_epochs = o2[:n_epochs * EPOCH_SAMPLES].reshape(n_epochs, EPOCH_SAMPLES)

    for i in range(n_epochs):
        feats = []
        for sig in [f4_epochs[i], c4_epochs[i], o2_epochs[i]]:
            feats.extend(bandpower(sig, SF_TARGET, BANDS))
            feats.extend(hjorth_parameters(sig))
        X.append(feats)
        y.append(1) 
        subject_ids.append(subj)

# === Save ===
X = np.array(X)
y = np.array(y)
subject_ids = np.array(subject_ids)

os.makedirs(RESULTS_PATH, exist_ok=True)
np.savez(OUTFILE, features=X, labels=y, subject_ids=subject_ids)

print("Saved:", OUTFILE)
print("Shape:", X.shape, y.shape, "Subjects:", len(np.unique(subject_ids)))