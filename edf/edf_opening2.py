import mne
import os

# Replace with your actual Downloads path, example for macOS
edf_path = os.path.expanduser("~/Downloads/ins1.edf")

# Load EDF file (preload=True loads data into memory)
raw = mne.io.read_raw_edf(edf_path, preload=True)

# Print info about the data
print(raw.info)

# Show channel names
print("Channels:", raw.ch_names)

# Plot the raw data (interactive plot)
raw.plot(duration=10, n_channels=10)