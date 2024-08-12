import numpy as np
from scipy.signal import get_window
import scipy.signal as signal
from scipy.fft import fft
import math
from scipy.signal import hilbert


# Estimate frequency and phase with DFT⁰ ((Rodriguez Paper)
def phase_estimation_DFT0(s_tone, Fs, N_DFT):

    window_type = 'hann'
    M = len(s_tone)
    window = get_window(window_type, M)
    s_tone = s_tone * window

    # Zero-pad the signal to length N_DFT
    s_tone_padded = np.pad(s_tone, (0, N_DFT - M), 'constant')

    # Compute the DFT
    X = fft(s_tone_padded, N_DFT)

    # Find the peak in the magnitude spectrum
    magnitude_spectrum = np.abs(X) # Magnitude of the DFT (Amplitude)
    k_max = np.argmax(magnitude_spectrum) # Maximum Amplitude
    f0_estimated = k_max*(Fs)/(N_DFT) # estimated frequency of the single tone

    # Estimate the phase
    phi0_estimated = np.angle(X[k_max]) # Argument (angle) of the DFT function

    return f0_estimated, phi0_estimated


# Estimate frequency with DFT¹ instantaneous estimation (Rodriguez Paper)
def freq_estimation_DFT1(s_tone, Fs, N_DFT):

    ###......Estimate the frequency......###
    window_type = 'hann'
    M = len(s_tone)
    
    # Get the window type
    window = get_window(window_type, M-1)
    
    # Calculate the approx. first derivative of single tone
    s_tone_diff = Fs * np.diff(s_tone)
    s_tone = s_tone[1:]

    # Windowing
    s_tone_windowed = s_tone * window
    s_tone_diff_windowed = s_tone_diff * window

    # Zero-Padding of the signal
    s_tone_padded = np.pad(s_tone_windowed, (0, N_DFT - M), 'constant')
    s_tone_padded_diff = np.pad(s_tone_diff_windowed, (0, N_DFT - M), 'constant')

    # Calculate the DFT
    X = fft(s_tone_padded,n=N_DFT)
    X_diff = fft(s_tone_padded_diff,n=N_DFT)

    # Compute the amplitude spectrum and max. amplitude
    abs_X = np.abs(X)
    k_max = np.argmax(abs_X)
    abs_X_diff = np.abs(X_diff)

    # Estimated frequency of the single tone
    F_kmax = (np.pi * k_max) / (N_DFT * np.sin(np.pi * k_max/ N_DFT))
    f0_estimated = (F_kmax * abs_X_diff[k_max]) / (2 * np.pi * abs_X[k_max])

     # Validate the frequency result
    k_DFT = (N_DFT * f0_estimated) / Fs
    try:
        k_DFT >= (k_max - 0.5) and k_DFT < (k_max + 0.5)
    except ValueError: print("estimated frequency is not valid")

    return f0_estimated


# Estimate phase with DFT¹ instantaneous estimation (Rodriguez Paper)
def phase_estimation_DFT1(s_tone, Fs, N_DFT, f0_estimated):

    ###......Estimate the frequency......###
    window_type = 'hann'
    M = len(s_tone)

    # Get the window type
    window = get_window(window_type, M-1)
    
    # Calculate the approx. first derivative of single tone
    s_tone_diff = Fs * np.diff(s_tone)

    # Windowing
    s_tone_diff_windowed = s_tone_diff * window

    # Zero-Padding of the signal
    s_tone_padded_diff = np.pad(s_tone_diff_windowed, (0, N_DFT - M), 'constant')

    # Calculate the DFT
    X_diff = fft(s_tone_padded_diff,n=N_DFT)

    k_DFT = (N_DFT * f0_estimated) / Fs

     # Validate the frequency result
    _, phi_DFT0 = phase_estimation_DFT0(s_tone, Fs, N_DFT) # Calculate phase with DFT⁰ method to compare the values

    omega_0 = 2*np.pi*f0_estimated/Fs
    k_low = math.floor(k_DFT)
    k_high = math.ceil(k_DFT)

    theta_low = np.angle(X_diff[k_low])
    theta_high = np.angle(X_diff[k_high])
    theta = (k_DFT - k_low) * (theta_high - theta_low) / (k_high - k_low) + theta_low

    numerator = np.tan(theta) * (1 - np.cos(omega_0)) + np.sin(omega_0)
    denominator = 1 - np.cos(omega_0) - np.tan(theta) * np.sin(omega_0)
    phase_estimated = np.arctan(numerator/denominator)
    
    # Calculate both possible values of phi and compare them
    phi_1 = phase_estimated 
    phi_2 = phase_estimated + np.pi if np.arctan(phase_estimated) >= 0 else phase_estimated - np.pi

    if abs(phi_1 - phi_DFT0) < abs(phi_2 - phi_DFT0):  # compare with phi calculated via DFT⁰
        phi = phi_1
    else:
        phi = phi_2

    return phi


###..... Estimate the instantaneous frequency of a tone in 
def segmented_freq_estimation_DFT0(s_in, f_s, num_cycles, N_DFT, nominal_enf):

    step_size = int(f_s // nominal_enf)  # samples per nominal enf cycle

    num_blocks = len(s_in) // step_size - (num_cycles - 1)

    segments = [s_in[i * step_size : (i + num_cycles) * step_size] for i in range(num_blocks)]

    freqs = []
    for i in range(len(segments)):
        freq,_ = phase_estimation_DFT0(segments[i], f_s, N_DFT)
        freqs.append(freq)

    freqs = np.array(freqs)
    
    return freqs


def segmented_freq_estimation_DFT1(s_in, f_s, num_cycles, N_DFT, nominal_enf):
    step_size = int(f_s // nominal_enf)  # samples per nominal enf cycle

    num_blocks = len(s_in) // step_size - (num_cycles - 1)
    
    segments = [s_in[i * step_size : (i + num_cycles) * step_size] for i in range(num_blocks)]

    freqs = []
    for i in range(len(segments)):
        freq = freq_estimation_DFT1(segments[i], f_s, N_DFT)
        freqs.append(freq)

    freqs = np.array(freqs)
    
    return freqs


def segmented_phase_estimation_DFT0(s_in, f_s, num_cycles, N_DFT, nominal_enf):
    step_size = int(f_s // nominal_enf)  # samples per nominal enf cycle

    num_blocks = len(s_in) // step_size - (num_cycles - 1)

    segments = [s_in[i * step_size : (i + num_cycles) * step_size] for i in range(num_blocks)]

    phases = []
    for segment in segments:
        _, phase = phase_estimation_DFT0(segment, f_s, N_DFT)
        phases.append(phase)

    phases = np.array(phases)
    phases = np.unwrap(phases)
    return phases


def segmented_phase_estimation_DFT1(s_in, f_s, num_cycles, N_DFT, nominal_enf):
    step_size = int(f_s // nominal_enf)  # samples per nominal enf cycle

    num_blocks = len(s_in) // step_size - (num_cycles - 1)

    segments = [s_in[i * step_size : (i + num_cycles) * step_size] for i in range(num_blocks)]

    phases = []
    for segment in segments:
        _, phase = phase_estimation_DFT1(segment, f_s, N_DFT)
        phases.append(phase)

    phases = np.array(phases)
    #phases = np.unwrap(phases)
    return phases


# Hilbert instantaneous freq estimation 
def hilbert_instantaneous_freq(signal, fs):
    analytic_sig = hilbert(signal)
    inst_phase  = np.unwrap(np.angle(analytic_sig))
    inst_freq = (np.diff(inst_phase)/(2.0*np.pi) * fs)
    inst_freq = np.append(inst_freq, inst_freq[-1]) # Diff reduces the number of results by 1 -> dulicate the last frequency
    return inst_freq

# Hilbert instantaneous phase estimation
def hilbert_instantaneous_phase(signal):
    analytic_sig = hilbert(signal)
    inst_phase  = np.unwrap(np.angle(analytic_sig))
    return inst_phase

# Hilbert segmented phase estimation
def segmented_phase_estimation_hilbert(s_in, f_s, num_cycles, nominal_enf):

    step_size = int(f_s // nominal_enf)

    num_blocks = len(s_in) // step_size - (num_cycles - 1)

    segments = [s_in[i * step_size : (i + num_cycles) * step_size] for i in range(num_blocks)]

    phases = []
    for i in range(len(segments)):
        phase = hilbert_instantaneous_phase(segments[i])
        phase = np.mean(phase)
        phases.append(phase)
    
    phases = np.unwrap(phases)
    phases = np.array(phases)

    return phases

# Instantaneous phase estimation via scipy
def scipy_IF_estimation(sig, fs):

    nperseg = 10 *fs
    freqs,times,stft = signal.stft(sig, fs=fs, nperseg=nperseg) # Apply STFT
    peak_freqs = [freqs[idx] for t in range(len(times)) if (idx := np.argmax(stft[:,t]))] # Extract peak for each point in time

    return peak_freqs

