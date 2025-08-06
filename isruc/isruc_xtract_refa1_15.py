import os
import numpy as np
from scipy.io import loadmat
from scipy.signal import welch
from scipy.integrate import simpson
from scipy.stats import skew
from scipy.signal import butter, filtfilt

# === Feature functions ===
def bandpower(data, sf, bands):
    freqs, psd = welch(data, sf, nperseg=sf * 2)
    return [
        simpson(psd[(freqs >= low) & (freqs <= high)],
                freqs[(freqs >= low) & (freqs <= high)])
        for low, high in bands
    ]

def band_ratios(powers):
    delta, theta, alpha, beta, _ = powers
    eps = 1e-10
    return [
        delta / (theta + eps),
        delta / (alpha + eps),
        alpha / (beta + eps),
        theta / (alpha + eps),
    ]

def hjorth_parameters(data):
    diff1 = np.diff(data)
    diff2 = np.diff(diff1)
    var_zero = np.var(data)
    var_d1 = np.var(diff1)
    var_d2 = np.var(diff2)
    mobility = np.sqrt(var_d1 / (var_zero + 1e-10))
    complexity = np.sqrt(var_d2 / (var_d1 + 1e-10)) / (mobility + 1e-10)
    return [var_zero, mobility, complexity]

def statistical_features(data):
    return [np.mean(data), np.std(data), skew(data)]

def bandpass_filter(data, sf, low=0.5, high=40):
    b, a = butter(N=4, Wn=[low, high], btype='bandpass', fs=sf)
    return filtfilt(b, a, data)

# === Config ===
BASE_PATH = os.path.join("..", "..", "..", "data", "ISRUC sleep", "data")
RESULTS_PATH = os.path.join("..", "..", "results")
OUTFILE = os.path.join(RESULTS_PATH, "isruc_18fts.npz")
CHANNELS = ["F4_A1", "C4_A1", "O2_A1"]
SF_TARGET = 200
EPOCH_LEN = 30
EPOCH_SAMPLES = SF_TARGET * EPOCH_LEN
BANDS = [(0.5, 4), (4, 8), (8, 12), (12, 30), (30, 50)]

X, y, subject_ids = [], [], []

def process_subject(subj_folder):
    mat_file = None
    for f in os.listdir(subj_folder):
        if f.endswith(".mat"):
            mat_file = os.path.join(subj_folder, f)
            break
    if not mat_file:
        return

    mat = loadmat(mat_file)
    try:
        raw_f4 = mat['F4_A1'].flatten()
        raw_c4 = mat['C4_A1'].flatten()
        raw_o2 = mat['O2_A1'].flatten()
    except KeyError:
        return

    orig_sf = mat.get('sampling_frequency', SF_TARGET)
    if isinstance(orig_sf, np.ndarray):
        orig_sf = int(orig_sf.flatten()[0])

    # Resample and filter
    def resample_and_filter(sig):
        duration = len(sig) / orig_sf
        from scipy.signal import resample
        resampled = resample(sig, int(duration * SF_TARGET))
        return bandpass_filter(resampled, SF_TARGET)

    f4 = resample_and_filter(raw_f4)
    c4 = resample_and_filter(raw_c4)
    o2 = resample_and_filter(raw_o2)

    min_len = min(len(f4), len(c4), len(o2))
    n_epochs = min_len // EPOCH_SAMPLES
    if n_epochs == 0:
        return

    f4_epochs = f4[:n_epochs * EPOCH_SAMPLES].reshape(n_epochs, EPOCH_SAMPLES)
    c4_epochs = c4[:n_epochs * EPOCH_SAMPLES].reshape(n_epochs, EPOCH_SAMPLES)
    o2_epochs = o2[:n_epochs * EPOCH_SAMPLES].reshape(n_epochs, EPOCH_SAMPLES)

    subject_id = os.path.basename(subj_folder)

    for i in range(n_epochs):
        feats = []
        for seg in [f4_epochs[i], c4_epochs[i], o2_epochs[i]]:
            psd_powers = bandpower(seg, SF_TARGET, BANDS)
            feats.extend(psd_powers)
            feats.extend(band_ratios(psd_powers))
            feats.extend(hjorth_parameters(seg))
            feats.extend(statistical_features(seg))
        X.append(feats)
        y.append(1)  # All ISRUC are labeled as disordered
        subject_ids.append(subject_id)

for subj in sorted(os.listdir(BASE_PATH)):
    full_path = os.path.join(BASE_PATH, subj)
    if os.path.isdir(full_path):
        print(f"Processing {subj}")
        process_subject(full_path)

os.makedirs(RESULTS_PATH, exist_ok=True)
np.savez(OUTFILE,
         features=np.array(X),
         labels=np.array(y),
         subject_ids=np.array(subject_ids))

print(f"Saved to: {OUTFILE}")
print(f"Shape: features = {np.array(X).shape}, labels = {np.array(y).shape}, subjects = {len(set(subject_ids))} unique")