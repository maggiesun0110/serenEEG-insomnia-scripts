import os
from scipy.io import loadmat
import mne

# === Paths ===
ISRUC_PATH = os.path.join("..", "..","data", "ISRUC sleep", "data")
SLEEPEDF_SC_PATH = os.path.join("..", "..","data", "SleepEDF", "sc")
SLEEPEDF_ST_PATH = os.path.join("..", "..","data", "SleepEDF", "st")
CAPSLEEP_PATH = os.path.join("..", "..", "data", "ins1.edf")
MENDELEY_INS_PATH = os.path.join("..", "..", "data", "Mendeley", "insomnia")
MENDELEY_NORM_PATH = os.path.join("..", "..", "data", "Mendeley", "normal")

# === ISRUC ===
print("=== ISRUC (first subject) ===")
for subj in sorted(os.listdir(ISRUC_PATH)):
    subj_path = os.path.join(ISRUC_PATH, subj)
    if not os.path.isdir(subj_path):
        continue
    mat_file = next((f for f in os.listdir(subj_path) if f.endswith(".mat")), None)
    if mat_file:
        mat_path = os.path.join(subj_path, mat_file)
        mat = loadmat(mat_path)
        channels_in_file = [key for key in mat.keys() if not key.startswith("__")]
        print(f"{subj}: {channels_in_file}")
    break

# === Sleep-EDF SC ===
print("\n=== Sleep-EDF SC (first file) ===")
for f in sorted(os.listdir(SLEEPEDF_SC_PATH)):
    if f.endswith(".edf"):
        edf_path = os.path.join(SLEEPEDF_SC_PATH, f)
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        print(f"{f}: {raw.ch_names}")
        break

# === Sleep-EDF ST ===
print("\n=== Sleep-EDF ST (first file) ===")
for f in sorted(os.listdir(SLEEPEDF_ST_PATH)):
    if f.endswith(".edf"):
        edf_path = os.path.join(SLEEPEDF_ST_PATH, f)
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        print(f"{f}: {raw.ch_names}")
        break

# === CAP Sleep ===
print("\n=== CAP Sleep ===")
cap_file = os.path.join("..", "..", "data", "ins1.edf")
if os.path.exists(cap_file):
    raw = mne.io.read_raw_edf(cap_file, preload=False, verbose=False)
    print(f"ins1.edf: {raw.ch_names}")
else:
    print("ins1.edf not found in data folder")

# === Mendeley Insomnia ===
print("\n=== Mendeley Insomnia (first file) ===")
for f in sorted(os.listdir(MENDELEY_INS_PATH)):
    if f.endswith(".edf"):
        edf_path = os.path.join(MENDELEY_INS_PATH, f)
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        print(f"{f}: {raw.ch_names}")
        break

# === Mendeley Normal ===
print("\n=== Mendeley Normal (first file) ===")
for f in sorted(os.listdir(MENDELEY_NORM_PATH)):
    if f.endswith(".edf"):
        edf_path = os.path.join(MENDELEY_NORM_PATH, f)
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        print(f"{f}: {raw.ch_names}")
        break