import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
from enfify.preprocessing import cut_signal
from enfify.synthetic_signals import func_ENF_synthesis_corrupted_harmonic

## Constants to modify
FUNDAMENTAL_ENF = 50  # ENF freq
HARMONIC_INDEX = np.array([1, 2, 3, 4, 5, 6])
CORRUPTED_INDEX = np.array([0, 0, 0])
DURATION = 40  # in seconds
FS = 1000  # sampling freq
NUMBER = 1000  # Number of signals

# Files paths to store and cut files
UNCUT_DIR = "/home/leo_dacasi/Dokumente/summerofcode/Enfify Data Synced/raw/synthetic/40s/uncut"
CUT_DIR = "/home/leo_dacasi/Dokumente/summerofcode/Enfify Data Synced/raw/synthetic/40s/cut"

# Cut variables
cut_len = [np.random.randint(1 * FS, 20 * FS) for i in range(NUMBER)]
cut_point = [np.random.randint(3 * FS, 17 * FS) for i in range(NUMBER)]

for i in tqdm(range(NUMBER)):

    # Generate 3 unique random integers between 2 and 6 (inclusive)
    random_numbers = np.random.choice(np.arange(2, 7), size=3, replace=False)
    CORRUPTED_INDEX[:] = random_numbers

    raw_sig, freqs, _ = func_ENF_synthesis_corrupted_harmonic(
        FUNDAMENTAL_ENF, HARMONIC_INDEX, CORRUPTED_INDEX, DURATION, FS, False
    )

    cut_sig = cut_signal(raw_sig, cut_point[i], cut_len[i])
    len_cut_sig = len(cut_sig)
    uncut_sig = raw_sig[:len_cut_sig]
    # cut_sig = cut_sig[cut_point[i] - 15 * FS : cut_point[i] + 15 * FS]
    # uncut_sig = raw_sig[cut_point[i] - 15 * FS : cut_point[i] + 15 * FS]
    # freqs = freqs[:len_cut_sig]

    wavfile.write(UNCUT_DIR + "/" + str(i) + "_audio.wav", FS, uncut_sig)
    # np.save(UNCUT_DIR + "/freqs/" + str(i) + "freqs.npy", freqs)
    wavfile.write(CUT_DIR + "/" + str(i) + "_cut_audio.wav", FS, cut_sig)
