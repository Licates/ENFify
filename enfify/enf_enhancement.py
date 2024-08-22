"""Module for ENF enhancement"""

# TODO: Add paper DOI links in docstrings of functions.

import cmath
import math

import numpy as np
from .enf_estimation import segmented_freq_estimation_DFT1
from scipy.fft import fft
from scipy.signal import windows
from tqdm import tqdm

# import scipy.signal as signal
# from scipy.signal import get_window
# from scipy.fft import fft


# ...................................Variational Mode Decomposition...................................#


def VariationalModeDecomposition(signal, alpha, tau, num_modes, enforce_DC, tolerance):
    """_summary_

    Args:
        signal (_type_): _description_
        alpha (_type_): _description_
        tau (_type_): _description_
        num_modes (_type_): _description_
        enforce_DC (_type_): _description_
        tolerance (_type_): _description_

    Returns:
        _type_: _description_
    """

    # Mirror signal at boundaries
    if len(signal) % 2:
        midpoint = math.ceil(len(signal) / 2)
        left_mirror = np.concatenate((np.flipud(signal[: midpoint - 1]), signal), axis=0)
        mirrored_signal = np.concatenate((left_mirror, np.flipud(signal[midpoint:])), axis=0)
    else:
        midpoint = len(signal) // 2
        left_mirror = np.concatenate((np.flipud(signal[:midpoint]), signal), axis=0)
        mirrored_signal = np.concatenate((left_mirror, np.flipud(signal[midpoint:])), axis=0)

    # Define time and frequency domains
    total_length = len(mirrored_signal)
    time_domain = np.arange(1, total_length + 1) / total_length
    spectral_domain = time_domain - 0.5 - (1 / total_length)

    # Iteration parameters
    max_iterations = 500
    mode_alphas = alpha * np.ones(num_modes)

    # FFT of the mirrored signal and preparation for iterations
    signal_spectrum = np.fft.fftshift(np.fft.fft(mirrored_signal))
    positive_spectrum = np.copy(signal_spectrum)
    positive_spectrum[: total_length // 2] = 0

    # Initialize frequency center estimates
    freq_centers = np.zeros((max_iterations, num_modes))
    for mode in range(num_modes):
        freq_centers[0, mode] = (0.5 / num_modes) * mode

    # Enforce DC mode if required
    if enforce_DC:
        freq_centers[0, 0] = 0

    # Initialize dual variables and other parameters
    dual_vars = np.zeros((max_iterations, len(spectral_domain)), dtype=complex)
    convergence_criteria = tolerance + np.spacing(1)
    iteration_count = 0
    mode_sum = 0
    mode_spectra = np.zeros((max_iterations, len(spectral_domain), num_modes), dtype=complex)

    # Main iterative update loop
    while convergence_criteria > tolerance and iteration_count < max_iterations - 1:
        mode_sum = (
            mode_spectra[iteration_count, :, num_modes - 1]
            + mode_sum
            - mode_spectra[iteration_count, :, 0]
        )

        mode_spectra[iteration_count + 1, :, 0] = (
            positive_spectrum - mode_sum - dual_vars[iteration_count, :] / 2
        ) / (1 + mode_alphas[0] * (spectral_domain - freq_centers[iteration_count, 0]) ** 2)

        if not enforce_DC:
            freq_centers[iteration_count + 1, 0] = np.dot(
                spectral_domain[total_length // 2 :],
                abs(mode_spectra[iteration_count + 1, total_length // 2 :, 0]) ** 2,
            ) / np.sum(abs(mode_spectra[iteration_count + 1, total_length // 2 :, 0]) ** 2)

        for mode in range(1, num_modes):
            mode_sum = (
                mode_spectra[iteration_count + 1, :, mode - 1]
                + mode_sum
                - mode_spectra[iteration_count, :, mode]
            )
            mode_spectra[iteration_count + 1, :, mode] = (
                positive_spectrum - mode_sum - dual_vars[iteration_count, :] / 2
            ) / (
                1
                + mode_alphas[mode] * (spectral_domain - freq_centers[iteration_count, mode]) ** 2
            )
            freq_centers[iteration_count + 1, mode] = np.dot(
                spectral_domain[total_length // 2 :],
                abs(mode_spectra[iteration_count + 1, total_length // 2 :, mode]) ** 2,
            ) / np.sum(abs(mode_spectra[iteration_count + 1, total_length // 2 :, mode]) ** 2)

        dual_vars[iteration_count + 1, :] = dual_vars[iteration_count, :] + tau * (
            np.sum(mode_spectra[iteration_count + 1, :, :], axis=1) - positive_spectrum
        )

        iteration_count += 1

        convergence_criteria = np.spacing(1)
        for mode in range(num_modes):
            convergence_criteria += (1 / total_length) * np.dot(
                (
                    mode_spectra[iteration_count, :, mode]
                    - mode_spectra[iteration_count - 1, :, mode]
                ),
                np.conj(
                    mode_spectra[iteration_count, :, mode]
                    - mode_spectra[iteration_count - 1, :, mode]
                ),
            )

        convergence_criteria = np.abs(convergence_criteria)

    # Postprocessing to extract modes and their spectra
    max_iterations = min(max_iterations, iteration_count)
    final_freq_centers = freq_centers[:max_iterations, :]

    half_idxs = np.flip(np.arange(1, total_length // 2 + 1), axis=0)
    final_mode_spectra = np.zeros((total_length, num_modes), dtype=complex)
    final_mode_spectra[total_length // 2 : total_length, :] = mode_spectra[
        max_iterations - 1, total_length // 2 : total_length, :
    ]
    final_mode_spectra[half_idxs, :] = np.conj(
        mode_spectra[max_iterations - 1, total_length // 2 : total_length, :]
    )
    final_mode_spectra[0, :] = np.conj(final_mode_spectra[-1, :])

    modes = np.zeros((num_modes, len(time_domain)))
    for mode in range(num_modes):
        modes[mode, :] = np.real(np.fft.ifft(np.fft.ifftshift(final_mode_spectra[:, mode])))

    modes = modes[:, total_length // 4 : 3 * total_length // 4]

    mode_spectra_final = np.zeros((modes.shape[1], num_modes), dtype=complex)
    for mode in range(num_modes):
        mode_spectra_final[:, mode] = np.fft.fftshift(np.fft.fft(modes[mode, :]))

    return modes, mode_spectra_final, final_freq_centers


# ...................................Maximum Likelyhood estimator...................................#


def stft_search(sig, fs, win_dur, step_dur, fc, bnd, fft_fac):
    win_len = int(win_dur * fs)
    win_func = windows.boxcar(win_len)
    step = int(step_dur * fs)
    NFFT = int(fft_fac * fs)

    win_pos = np.arange(0, len(sig) - win_len + 1, step)
    IF = np.zeros(len(win_pos))

    # Set search region
    search_1st = np.arange(round((50 - bnd[0] / 2) * fft_fac), round((50 + bnd[0] / 2) * fft_fac))
    len_band = len(search_1st)
    search_reg = np.kron(search_1st, (fc / 50)).astype(int)

    # Search loop
    for i, pos in enumerate(win_pos):
        temp_fft = fft(sig[pos : pos + win_len] * win_func, n=NFFT)
        half_fft = temp_fft[: NFFT // 2]
        abs_half_fft = np.abs(half_fft)

        fbin_cand = abs_half_fft[search_reg]
        fbin_cand = fbin_cand.reshape((len(fc), len_band))
        weighted_fbin = np.sum(fbin_cand**2, axis=0)
        max_val = np.max(weighted_fbin)
        peak_loc = search_1st[np.where(weighted_fbin == max_val)[0][0]]
        IF[i] = peak_loc * fs / NFFT * 2

    norm_fc = 100
    norm_bnd = 100 * bnd[0] / fc[0]
    IF[IF < norm_fc - norm_bnd] = norm_fc - norm_bnd
    IF[IF > norm_fc + norm_bnd] = norm_fc + norm_bnd

    return IF
