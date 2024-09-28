import numpy as np
import os
from glob import glob
import yaml
from scipy.io import wavfile
from tqdm import tqdm
from pathlib import Path
from enfify.config import ENFIFY_DIR
from enfify.pipeline import freq_CNN_feature_pipeline

PROCESSED_DATA_DIR = Path("/home/leo_dacasi/Dokumente/summerofcode/Enfify Data Synced/processed")
INTERIM_DATA_DIR = Path(
    "/home/leo_dacasi/Dokumente/summerofcode/Enfify Data Synced/interim/Carioca/Carioca_10s"
)

# Paths
FEATURE_PATH = PROCESSED_DATA_DIR / "Carioca_10s_CNN_BiLSTM"
os.makedirs(FEATURE_PATH, exist_ok=True)

# Get Cut and Uncut Files
filenames = sorted(glob(str(Path(INTERIM_DATA_DIR / "*.wav"))))
cut_files = [f for f in filenames if "tamp" in f]
uncut_files = [f for f in filenames if "auth" in f]

cut_freqs = []
uncut_freqs = []


for file in tqdm(cut_files):
    basename = os.path.splitext(os.path.basename(file))[0]
    sample_freq, sig = wavfile.read(file)
    with open(ENFIFY_DIR / "config_nature.yml", "r") as f:
        config = yaml.safe_load(f)
    cut_freq = freq_CNN_feature_pipeline(sig, sample_freq, config)
    cut_freqs.append(cut_freq)
    print(len(cut_freq))


for file in tqdm(uncut_files):
    basename = os.path.splitext(os.path.basename(file))[0]
    sample_freq, sig = wavfile.read(file)
    with open(ENFIFY_DIR / "config_nature.yml", "r") as f:
        config = yaml.safe_load(f)
    uncut_freq = freq_CNN_feature_pipeline(sig, sample_freq, config)
    uncut_freqs.append(uncut_freq)
    print(len(uncut_freq))

# Label Combine and shuffle the Data
labels = np.concatenate([np.ones(len(cut_freqs)), np.zeros(len(uncut_freqs))])
freq_data = cut_freqs + uncut_freqs
indices = np.random.permutation(len(freq_data))

freqs_data = np.array([freq_data[i] for i in indices], dtype=object)
labels = labels[indices]

# Save the data
np.save(FEATURE_PATH + "/" + "carioca_freqs_10s.npy", freqs_data)
np.save(FEATURE_PATH + "/" + "carioca_freqs_10s_labels.npy", labels)
