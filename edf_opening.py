import mne
import os
import matplotlib.pyplot as plt

# === Set file paths ===
base_dir = os.path.join('..', '..', 'data', 'SleepEDF', "SC")
psg_file = os.path.join(base_dir, 'SC4001E0-PSG.edf')
hypnogram_file = os.path.join(base_dir, 'SC4001EC-Hypnogram.edf')

# === Load raw EEG data ===
print("Loading EEG file...")
raw = mne.io.read_raw_edf(psg_file, preload=True)

# Pick only the Fpz-Cz channel (closest to F3-A2)
raw.pick_channels(['EEG Fpz-Cz'])

# === Load and set hypnogram annotations ===
print("Loading hypnogram annotations...")
annotations = mne.read_annotations(hypnogram_file)
raw.set_annotations(annotations)

# === Print info ===
print("\n=== EEG Info ===")
print(raw.info)

print("\n=== Hypnogram Annotations ===")
for ann in annotations:
    print(f"{ann['onset']}s — {ann['duration']}s: {ann['description']}")

# === Plot EEG with sleep stages ===
print("\nOpening EEG plot window...")
raw.plot(duration=60, n_channels=1, title="Fpz-Cz EEG with Sleep Stages")
plt.show()