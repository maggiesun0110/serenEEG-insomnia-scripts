import os
import mne
import numpy as np
from scipy.signal import welch, resample
from mne.io import read_raw_edf

# --- Config ---
data_root = "../../data/SleepEDF"
output_path = "../results/features.npz"
log_path = "../results/failures.log"
segment_duration_sec = 30
resample_rate = 200

# --- Frequency bands ---
def compute_psd_bands(segment, sf):
    freqs, psd = welch(segment, sf, nperseg=sf*2)
    band_limits = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 12), 'beta': (12, 30), 'gamma': (30, 45)}
    band_powers = []
    for band in band_limits.values():
        idx = np.logical_and(freqs >= band[0], freqs <= band[1])
        band_power = np.mean(psd[idx])
        band_powers.append(band_power)
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

for label, subdir in enumerate(["SC", "ST"]):
    folder = os.path.join(data_root, subdir)
    for file in os.listdir(folder):
        if not file.endswith(".edf") or "Hypnogram" in file:
            continue
        try:
            subject_id = file.split("-")[0]  # SC4001E0-PSG.edf -> SC4001E0
            prefix = subject_id[:-2]         # Remove E0/F0/G0 suffix
            full_path = os.path.join(folder, file)
            raw = read_raw_edf(full_path, preload=True, verbose=False)

            if "EEG Fpz-Cz" not in raw.ch_names:
                raise ValueError("Channel EEG Fpz-Cz not found")

            eeg = raw.copy().pick_channels(["EEG Fpz-Cz"])
            eeg_data = eeg.get_data()[0]  # shape (n_samples,)
            original_sf = int(eeg.info['sfreq'])
            target_n_samples = int(len(eeg_data) * resample_rate / original_sf)
            eeg_data = resample(eeg_data, target_n_samples)

            total_samples = len(eeg_data)
            segment_len = segment_duration_sec * resample_rate  # 6000
            n_segments = total_samples // segment_len

            for i in range(n_segments):
                segment = eeg_data[i * segment_len:(i + 1) * segment_len]
                if len(segment) < segment_len:
                    continue
                psd_features = compute_psd_bands(segment, resample_rate)
                hjorth_feats = hjorth_parameters(segment)
                full_feature = psd_features + hjorth_feats
                features.append(full_feature)
                labels.append(label)  # 0 for SC (healthy), 1 for ST (insomnia)

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