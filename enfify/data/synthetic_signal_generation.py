import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
from enfify.preprocessing import cut_signal
from enfify.synthetic_signals import func_ENF_synthesis_corrupted_harmonic

## Constants to modify
FUNDAMENTAL_ENF = 50  # ENF freq
HARMONIC_INDEX = np.array([1, 2, 3, 4, 5, 6])
CORRUPTED_INDEX = np.array([0, 0, 0])
DURATION = 10  # in seconds
FS = 1000  # sampling freq
NUMBER = 2000  # Number of signals

# Files paths to store and cut files
DIR = (
    "/home/leo_dacasi/Dokumente/summerofcode/Enfify Data Synced/interim/Synthetic/10s_cut_variety"
)

# Cut variables
cut_len = np.array([np.random.randint(2, 19) for i in range(NUMBER)])

cut_point = np.array([np.random.randint(2 * FS, 8 * FS) for i in range(NUMBER)])

for i in tqdm(range(NUMBER)):

    # Generate 3 unique random integers between 2 and 6 (inclusive)
    random_numbers = np.random.choice(np.arange(3, 7), size=3, replace=False)
    CORRUPTED_INDEX[:] = random_numbers

    NEW_DURATION = DURATION + cut_len[i]
    raw_sig, _, _ = func_ENF_synthesis_corrupted_harmonic(
        FUNDAMENTAL_ENF, HARMONIC_INDEX, CORRUPTED_INDEX, NEW_DURATION, FS, False
    )

    cut_sig = cut_signal(raw_sig, cut_point[i], cut_len[i])
    uncut_sig = raw_sig[: -cut_len[i]]

    wavfile.write(DIR + "/" + str(i) + "_auth_audio.wav", FS, uncut_sig)
    wavfile.write(DIR + "/" + str(i) + "_tamp_audio.wav", FS, cut_sig)
