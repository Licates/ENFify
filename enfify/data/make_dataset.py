"""Script to create the dataset."""

import io
import os

import numpy as np
import requests
from scipy.io import wavfile
from enfify.preprocessing import downsampling_alpha
from enfify.synthetic_signals import random_signal

np.random.seed(42)


def download_whu_01():
    url = "https://github.com/ghua-ac/ENF-WHU-Dataset/raw/78ed7f3784949f769f291fc1cb94acd10da6322f/ENF-WHU-Dataset/H0/01.wav"

    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(f"Failed to download file. Status code: {response.status_code}")

    file_bytes = io.BytesIO(response.content)

    sample_freq, sig = wavfile.read(file_bytes)

    return sig, sample_freq


def make_whu_sample_files():
    uncut_path = "samples/whu_uncut_min_001_ref.wav"
    cut_path = "samples/whu_cut_min_001_ref.wav"

    # downloading
    sig, sample_freq = download_whu_01()
    downsampling_alpha(sig, sample_freq, 1_000)
    sig = sig[: sample_freq * 60]

    # Save uncut file
    wavfile.write(uncut_path, sample_freq, sig)

    # cutting
    location = np.random.randint(0, len(sig) - sample_freq * 10)
    # location = (len(sig) - sample_freq * 10) // 2
    sig = sig[location : location + sample_freq * 10]

    # Save cut file
    wavfile.write(cut_path, sample_freq, sig)


def make_synthetic_sample_files():
    uncut_path = "samples/synthetic_uncut_0.wav"
    cut_path = "samples/synthetic_cut_0.wav"

    sig_uncut, sig_cut = random_signal(1, 60, 1_000, 50, 0.2, 10_000)

    wavfile.write(uncut_path, 1_000, sig_uncut)
    wavfile.write(cut_path, 1_000, sig_cut)


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

    make_whu_sample_files()

    make_synthetic_sample_files()
