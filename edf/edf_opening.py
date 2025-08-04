import mne
import os
import matplotlib.pyplot as plt

# === Set file paths ===
base_dir = os.path.join('..', '..', 'data', 'mendeley', "normal")
base_dir2 = os.path.join('..', '..', 'data', 'mendeley', 'insomnia')
psg_file = os.path.join(base_dir, 'Normal_Subject_01.edf')
psg_file2 = os.path.join(base_dir2, 'Raw_Signal_Psycophysiological_Insomnia_01.edf')

# === Load raw EEG data ===
print("Loading EEG file...")
raw = mne.io.read_raw_edf(psg_file, preload=True)

print("Loading EEG file...")
raw2 = mne.io.read_raw_edf(psg_file2, preload = True)

# # Pick only the Fpz-Cz channel (closest to F3-A2)

# === Print info ===
print("\n=== EEG Info ===")
print(raw.info)

print("\n==EEG info 2==")
print(raw2.info)

# === Plot EEG with sleep stages ===
print("\nOpening EEG plot window...")
raw.plot(duration=60, n_channels=1, title="normal")
plt.show()

raw2.plot(duration = 60, n_channels = 1, title = "insomnia")
plt.show()