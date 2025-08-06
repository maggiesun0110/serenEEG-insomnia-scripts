import os
import numpy as np
from scipy.io import loadmat
from scipy.signal import welch, butter, filtfilt, resample

# === Feature Functions (Mendeley-style) ===
def compute_mobility(data):
    diff1 = np.diff(data)
    var_zero = np.var(data)
    var_d1 = np.var(diff1)
    return np.sqrt(var_d1 / var_zero) if var_zero > 0 else 0

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
    mobility = np.sqrt(var_d1 / var_zero) if var_zero > 0 else 0
    complexity = np.sqrt(var_d2 / var_d1) / mobility if var_d1 > 0 and mobility > 0 else 0
    return complexity

# === Signal Preprocessing ===
def bandpass_filter(data, sf, low=0.5, high=40):
    b, a = butter(N=4, Wn=[low, high], btype='bandpass', fs=sf)
    return filtfilt(b, a, data)

def preprocess_signal(data, orig_sf, target_sf):
    filtered = bandpass_filter(data, orig_sf)
    if orig_sf != target_sf:
        duration_sec = len(filtered) / orig_sf
        resampled = resample(filtered, int(duration_sec * target_sf))
        return resampled
    return filtered

# === Configuration ===
BASE_PATH = os.path.join("..", "..", "..", "data", "ISRUC sleep", "data")
RESULTS_PATH = os.path.join("..", "..", "results")
CHANNELS = ["F4_A1", "C4_A1", "O2_A1"]
EPOCH_LEN = 30  # seconds
SF_TARGET = 200
EPOCH_SAMPLES = EPOCH_LEN * SF_TARGET

X, y, subject_ids = [], [], []

def extract_features_for_epoch(f4, c4, o2):
    feats = []
    # F4A1
    feats.append(compute_mobility(f4))
    feats.append(compute_alpha_beta_ratio(f4, SF_TARGET))
    # C4A1
    feats.append(compute_mobility(c4))
    feats.append(compute_alpha_beta_ratio(c4, SF_TARGET))
    # O2A1
    feats.append(compute_mobility(o2))
    feats.append(compute_alpha_beta_ratio(o2, SF_TARGET))
    feats.append(compute_mean(o2))
    feats.append(compute_std(o2))
    feats.append(compute_complexity(o2))
    return feats

def process_subject(subj_folder):
    mat_file = None
    for f in os.listdir(subj_folder):
        if f.endswith(".mat"):
            mat_file = os.path.join(subj_folder, f)
            break
    if not mat_file:
        print(f"No .mat file in {subj_folder}")
        return

    mat = loadmat(mat_file)
    try:
        raw_f4 = mat['F4_A1'].flatten()
        raw_c4 = mat['C4_A1'].flatten()
        raw_o2 = mat['O2_A1'].flatten()
    except KeyError as e:
        print(f"Missing channel in {subj_folder}: {e}")
        return

    orig_sf = mat.get('sampling_frequency', SF_TARGET)
    if isinstance(orig_sf, np.ndarray):
        orig_sf = int(orig_sf.flatten()[0])

    f4 = preprocess_signal(raw_f4, orig_sf, SF_TARGET)
    c4 = preprocess_signal(raw_c4, orig_sf, SF_TARGET)
    o2 = preprocess_signal(raw_o2, orig_sf, SF_TARGET)

    min_len = min(len(f4), len(c4), len(o2))
    n_epochs = min_len // EPOCH_SAMPLES
    if n_epochs == 0:
        print(f"Too short for 30s epochs: {subj_folder}")
        return

    f4_epochs = f4[:n_epochs * EPOCH_SAMPLES].reshape(n_epochs, EPOCH_SAMPLES)
    c4_epochs = c4[:n_epochs * EPOCH_SAMPLES].reshape(n_epochs, EPOCH_SAMPLES)
    o2_epochs = o2[:n_epochs * EPOCH_SAMPLES].reshape(n_epochs, EPOCH_SAMPLES)

    for i in range(n_epochs):
        feats = extract_features_for_epoch(f4_epochs[i], c4_epochs[i], o2_epochs[i])
        X.append(feats)
        y.append(1)  # All ISRUC here are disordered
        subject_ids.append(os.path.basename(subj_folder))

# === Run the extraction ===
for subj in sorted(os.listdir(BASE_PATH)):
    full_path = os.path.join(BASE_PATH, subj)
    if os.path.isdir(full_path):
        print(f"Processing {subj}")
        process_subject(full_path)

X = np.array(X)
y = np.array(y)
subject_ids = np.array(subject_ids)

print("Final shapes:")
print("X:", X.shape)
print("y:", y.shape)
print("Subjects:", len(np.unique(subject_ids)))

os.makedirs(RESULTS_PATH, exist_ok=True)
np.savez(os.path.join(RESULTS_PATH, "isruc_9features.npz"), features=X, labels=y, subject_ids=subject_ids)