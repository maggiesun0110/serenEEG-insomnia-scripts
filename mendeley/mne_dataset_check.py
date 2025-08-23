from mne.datasets import sample
from mne.io import read_raw_fif

# Download and get the sample data path
sample_data_folder = sample.data_path()
sample_data_raw_file = sample_data_folder / 'MEG' / 'sample' / 'sample_audvis_raw.fif'

# Load the raw file
raw = read_raw_fif(sample_data_raw_file, preload=True)

# Print info to see channels and samples
print(raw.info)