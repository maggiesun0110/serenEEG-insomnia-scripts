import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import welch

print("Current working directory:", os.getcwd())

# Base path for data folder (one level up + data)
base_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'data'))
print("Data base path:", base_path)

# Base path for saving results
results_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'results', 'batches'))
print("Results save path:", results_path)

all_subject_files = [f'subject{str(i).zfill(3)}' for i in range(76, 101)]

batch_size = 25
batches = [all_subject_files[i:i+batch_size] for i in range(0, len(all_subject_files), batch_size)]


def hjorth_params(epoch):
    first_deriv = np.diff(epoch)
    second_deriv = np.diff(first_deriv)
    activity = np.var(epoch)
    mobility = np.sqrt(np.var(first_deriv) / activity)
    complexity = np.sqrt(np.var(second_deriv) / np.var(first_deriv)) / mobility
    return activity, mobility, complexity


def extract_features(epoch, fs=200):
    frequencies, psd = welch(epoch, fs=fs, nperseg=fs * 2)
    delta_power = np.sum(psd[(frequencies >= 0.5) & (frequencies < 4)])
    theta_power = np.sum(psd[(frequencies >= 4) & (frequencies < 8)])
    alpha_power = np.sum(psd[(frequencies >= 8) & (frequencies < 13)])
    beta_power = np.sum(psd[(frequencies >= 13) & (frequencies < 30)])
    activity, mobility, complexity = hjorth_params(epoch)
    variance = np.var(epoch)
    return [delta_power, theta_power, alpha_power, beta_power, activity, mobility, complexity, variance]


def extract_features_for_subject(eeg_data):
    return np.array([extract_features(epoch) for epoch in eeg_data])


def get_mat_filepath(subject_folder):
    # subject_folder is full path to folder, so just join the .mat filename
    subj_num = os.path.basename(subject_folder).replace('subject', '')  # e.g. '002'
    mat_filename = f"subject{str(int(subj_num)).zfill(2)}.mat"  # e.g. 'subject02.mat'
    return os.path.join(subject_folder, mat_filename)


def get_labels_for_subject(subject_folder, eeg_length):
    label_txt = None
    for f in os.listdir(subject_folder):
        if 'labels_1.txt' in f:
            label_txt = os.path.join(subject_folder, f)
            break

    if label_txt is None:
        print(f"No labels_1.txt found in {subject_folder}. Skipping.")
        return None

    labels = np.loadtxt(label_txt)

    if len(labels) >= eeg_length:
        return labels[:eeg_length]
    else:
        print(f"Warning: {subject_folder} labels shorter than EEG. Skipping.")
        return None


def process_batch(subject_folders, batch_num):
    all_features = []
    all_labels = []

    for subject_folder_name in subject_folders:
        subject_folder = os.path.join(base_path, subject_folder_name)  # full path
        mat_path = get_mat_filepath(subject_folder)

        print(f"Looking for mat file: {mat_path}")
        if not os.path.exists(mat_path):
            print(f"{mat_path} not found. Skipping {subject_folder_name}.")
            continue

        mat_data = loadmat(mat_path)
        eeg_data = mat_data['F3_A2']

        print("EEG Data shape:", mat_data['F3_A2'].shape)
        print("EEG Data type:", type(mat_data['F3_A2']))

        print(f"{subject_folder}: EEG epochs = {eeg_data.shape[0]}")

        features = extract_features_for_subject(eeg_data)
        labels = get_labels_for_subject(subject_folder, eeg_data.shape[0])

        if labels is None:
            print(f"Skipping {subject_folder_name} due to missing or invalid labels.")
            continue

        print(f"{subject_folder}: Labels length = {len(labels)}")
        
        if len(labels) != features.shape[0]:
            print(f"Warning: Mismatch after trimming in {subject_folder_name}. Skipping.")
            continue

        all_features.append(features)
        all_labels.append(labels)

    if not all_features:
        print(f"No valid data to save in batch {batch_num}. Skipping.")
        return

    batch_features = np.vstack(all_features)
    batch_labels = np.hstack(all_labels)

    os.makedirs(results_path, exist_ok=True)

    np.save(os.path.join(results_path, f'features_batch_{batch_num}.npy'), batch_features)
    np.save(os.path.join(results_path, f'labels_batch_{batch_num}.npy'), batch_labels)

    print(f"Batch {batch_num} processed: features shape {batch_features.shape}, labels shape {batch_labels.shape}")


for i, batch_files in enumerate(batches, start=4):
    process_batch(batch_files, i)

# Sanity check (load saved files from results path)
features = np.load(os.path.join(results_path, 'features_batch_4.npy'))
labels = np.load(os.path.join(results_path, 'labels_batch_4.npy'))

print("Sanity Check:")
print("Features shape:", features.shape)
print("Labels shape:", labels.shape)