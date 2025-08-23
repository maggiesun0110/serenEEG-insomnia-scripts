import os
import mne
import numpy as np
from scipy.signal import welch, resample
from scipy.stats import kurtosis, skew
from mne.io import read_raw_edf
from numpy import log2

# --- Config ---
data_root = "../../data/mendeley"
output_path = "../results/features_a1_advanced_all_with_ids.npz"
log_path = "../results/features_a1_advanced_all_failures.log"
segment_duration_sec = 30
resample_rate = 200
channel = "A1"

# --- Frequency bands ---
band_limits = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (12, 30),
    'gamma': (30, 45)
}

def compute_psd_bands(segment, sf):
    freqs, psd = welch(segment, sf, nperseg=sf*2)
    powers = [np.mean(psd[(freqs >= low) & (freqs <= high)]) for low, high in band_limits.values()]
    return powers

def compute_band_ratios(powers):
    delta, theta, alpha, beta, gamma = powers
    eps = 1e-8  # avoid div zero
    ratios = [
        theta / (alpha + eps),             # theta/alpha
        delta / (beta + eps),              # delta/beta
        alpha / (beta + eps),              # alpha/beta
        (theta + alpha) / (beta + eps),   # (theta + alpha)/beta
        delta / (theta + eps)              # delta/theta
    ]
    return ratios

# --- Hjorth Parameters ---
def hjorth_parameters(segment):
    first_deriv = np.diff(segment)
    second_deriv = np.diff(first_deriv)
    var_zero = np.var(segment)
    var_d1 = np.var(first_deriv)
    var_d2 = np.var(second_deriv)
    activity = var_zero
    mobility = np.sqrt(var_d1 / var_zero) if var_zero != 0 else 0
    complexity = np.sqrt(var_d2 / var_d1) / mobility if var_d1 != 0 and mobility != 0 else 0
    return [activity, mobility, complexity]

# --- Spectral Entropy ---
def spectral_entropy(segment, sf):
    freqs, psd = welch(segment, sf, nperseg=sf*2)
    psd_norm = psd / np.sum(psd)
    entropy = -np.sum(psd_norm * log2(psd_norm + 1e-8))
    return [entropy]

# --- Statistical Features ---
def statistical_features(segment):
    rms = np.sqrt(np.mean(segment**2))
    zcr = ((segment[:-1] * segment[1:]) < 0).sum() / len(segment)
    return [skew(segment), kurtosis(segment), rms, zcr]

# --- Main ---
features = []
labels = []
subject_ids = []
failed_subjects = []

for folder_name, label in [("normal", 0), ("insomnia", 1)]:
    folder_path = os.path.join(data_root, folder_name)
    if not os.path.isdir(folder_path):
        print(f"⚠️ Folder not found: {folder_path}")
        continue

    for file in os.listdir(folder_path):
        if not file.endswith(".edf"):
            continue

        try:
            full_path = os.path.join(folder_path, file)
            raw = read_raw_edf(full_path, preload=True, verbose=False)

            # --- Filtering ---
            raw.filter(l_freq=0.5, h_freq=45, fir_design='firwin')
            raw.notch_filter(freqs=[50], fir_design='firwin')  # or 60Hz if in US

            if channel not in raw.ch_names:
                raise ValueError(f"Channel '{channel}' not found in {file}")

            eeg_data = raw.copy().pick_channels([channel]).get_data()[0]
            orig_sf = int(raw.info['sfreq'])
            eeg_data = resample(eeg_data, int(len(eeg_data) * resample_rate / orig_sf))

            segment_len = segment_duration_sec * resample_rate
            n_segments = len(eeg_data) // segment_len

            # Extract subject ID from filename (assumes unique per subject)
            subject_id = os.path.splitext(file)[0]

            for i in range(n_segments):
                segment = eeg_data[i * segment_len:(i + 1) * segment_len]
                if len(segment) < segment_len:
                    continue

                psd = compute_psd_bands(segment, resample_rate)
                hjorth = hjorth_parameters(segment)
                entropy = spectral_entropy(segment, resample_rate)
                stats = statistical_features(segment)
                ratios = compute_band_ratios(psd)

                full_feature = psd + hjorth + entropy + stats + ratios  # 18 features total
                features.append(full_feature)
                labels.append(label)
                subject_ids.append(subject_id)

        except Exception as e:
            failed_subjects.append((file, str(e)))
            continue

# --- Save ---
features = np.array(features)
labels = np.array(labels)
subject_ids = np.array(subject_ids)

np.savez(output_path, X=features, y=labels, subject_ids=subject_ids)

with open(log_path, 'w') as f:
    for subject, err in failed_subjects:
        f.write(f"{subject}: {err}\n")

print(f"✅ Saved {len(features)} samples to {output_path}")
print(f"❌ Failed subjects logged in {log_path}")