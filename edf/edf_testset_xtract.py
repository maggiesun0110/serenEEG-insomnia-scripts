import os
import mne
import numpy as np
from scipy.signal import welch, resample
from mne.io import read_raw_edf

# --- Config ---
data_root = "../../data/mendeley"
output_path = "../results/testset_features.npz"
log_path = "../results/testset_failures.log"
segment_duration_sec = 30
resample_rate = 200
channel_name = "F3A2"

# --- Frequency bands ---
def compute_psd_bands(segment, sf):
    freqs, psd = welch(segment, sf, nperseg=sf*2)
    band_limits = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 12), 'beta': (12, 30), 'gamma': (30, 45)}
    band_powers = [np.mean(psd[(freqs >= low) & (freqs <= high)]) for low, high in band_limits.values()]
    return band_powers

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

# --- Process each subject ---
features = []
labels = []
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

            if channel_name not in raw.ch_names:
                raise ValueError(f"Channel '{channel_name}' not found")

            eeg = raw.copy().pick_channels([channel_name])
            eeg_data = eeg.get_data()[0]  # shape (n_samples,)
            original_sf = int(eeg.info['sfreq'])
            eeg_data = resample(eeg_data, int(len(eeg_data) * resample_rate / original_sf))

            segment_len = segment_duration_sec * resample_rate
            n_segments = len(eeg_data) // segment_len

            for i in range(n_segments):
                segment = eeg_data[i * segment_len:(i + 1) * segment_len]
                if len(segment) < segment_len:
                    continue
                psd_features = compute_psd_bands(segment, resample_rate)
                hjorth_feats = hjorth_parameters(segment)
                full_feature = psd_features + hjorth_feats
                features.append(full_feature)
                labels.append(label)

        except Exception as e:
            failed_subjects.append((file, str(e)))
            continue

# --- Save ---
features = np.array(features)
labels = np.array(labels)
np.savez(output_path, X=features, y=labels)

# --- Log failures ---
with open(log_path, 'w') as f:
    for subject, err in failed_subjects:
        f.write(f"{subject}: {err}\n")

print(f"✅ Saved {len(features)} samples to {output_path}")
print(f"❌ Failed subjects logged in {log_path}")