import wfdb
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, freqz, tf2zpk, group_delay, lfilter
import numpy as np
from scipy.signal import find_peaks, convolve

#################################################################################################################################
#####################################>> load ECG Signal <<#######################################################################
def getSignal(path):
    record = wfdb.rdrecord(path)    # -> load data

    ecg_signal = record.p_signal[:, 0]  # -> get ECG signal
    fs = record.fs      # -> get Sampling frequency
    print(f"Sampling Frequency (fs): {fs} Hz")

    t = [i / fs for i in range(len(ecg_signal))]    # -> compute the time scale

    # ploting the first 5 seconds
    plt.plot(t[:fs * 5], ecg_signal[:fs * 5])
    plt.title("ECG Signal - First 5 Seconds")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (mV)")
    plt.grid(True)
    plt.show()

    return ecg_signal, fs
#################################################################################################################################
#################################################################################################################################


#################################################################################################################################
#########################################>> Filter ECG Signal <<#################################################################
# ----- Lowpass Filter  -----
def lowpass_filter(sig):
    b = [1, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 1]  # Based on Pan-Tompkins paper
    a = [1, -2, 1]
    return lfilter(b, a, sig)

# ----- Highpass Filter -----
def highpass_filter(sig):
    b = np.zeros(33); b[0] = -1; b[16] = 32; b[32] = 1
    a = [1, 1]
    return lfilter(b, a, sig)

#################################################################################################################################
#################################################################################################################################


#################################################################################################################################
#########################################>> Differentiate ECG Signal <<##########################################################

def derivative(sig):
    b = np.array([-1, -2, 0, 2, 1]) * (1/8)
    return lfilter(b, 1, sig)

#################################################################################################################################
#################################################################################################################################

#################################################################################################################################
#########################################>> Square ECG Signal <<#################################################################

def squaring(signal):
    squared = signal * signal
    return squared

#################################################################################################################################
#################################################################################################################################

#################################################################################################################################
#########################################>> integrate ECG Signal <<##############################################################

def moving_window_integration(signal, window_size):
    window = np.ones(window_size) / window_size
    integrated_signal = np.convolve(signal, window, mode='same')
    return integrated_signal

#################################################################################################################################
#################################################################################################################################


#################################################################################################################################
#########################################>> Detect ECG Signal Peaks<<############################################################

# -> Pan-Tompkins Peak Detection
def detect_peaks(signal, fs):
    min_distance = int(0.2 * fs)

    peaks, _ = find_peaks(signal, distance=min_distance)

    SPKI = 0.0
    NPKI = 0.0
    THRESHOLD_I1 = 0.0

    qrs_peaks = []

    for peak in peaks:
        PEAKI = signal[peak]

        if PEAKI > THRESHOLD_I1:
            qrs_peaks.append(peak)
            SPKI = 0.125 * PEAKI + 0.875 * SPKI
        else:
            NPKI = 0.125 * PEAKI + 0.875 * NPKI

        THRESHOLD_I1 = NPKI + 0.25 * (SPKI - NPKI)

    return qrs_peaks, THRESHOLD_I1



# -> LMS Adaptive Threshold Peak Detection
def lms_peak_detection(signal, fs, mu=0.02, initial_threshold_ratio=0.25, search_window_ms=100):
    signal = np.asarray(signal)
    initial_peak = np.max(signal[:fs]) 
    threshold = initial_threshold_ratio * initial_peak

    min_distance = int(0.2 * fs)
    search_window = int(search_window_ms * fs / 1000)

    peaks = []
    thresholds = []

    i = 0
    last_peak = -2 * min_distance

    while i < len(signal) - search_window:
        y = signal[i]

        if (i - last_peak) > min_distance and y > threshold:
            window_end = min(i + search_window, len(signal))
            local_max_index = np.argmax(signal[i:window_end]) + i
            peaks.append(local_max_index)
            last_peak = local_max_index

            error = signal[local_max_index] - threshold
            threshold += mu * error
            threshold = np.clip(threshold, 0.05 * initial_peak, 0.8 * initial_peak)

            i = local_max_index + 1
        else:
            error = 0
            threshold += mu * error
            threshold = np.clip(threshold, 0.05 * initial_peak, 0.8 * initial_peak)
            i += 1

        thresholds.append(threshold)

    return peaks, thresholds

#################################################################################################################################
#################################################################################################################################


#################################################################################################################################
#########################################>> Evaluate Detected Peaks <<###########################################################

def evaluate_peaks(detected_peaks, true_peaks, fs, tolerance=0.15):
    tolerance_samples = int(tolerance * fs)
    TP = 0
    FN = 0
    FP = 0

    matched = np.zeros(len(true_peaks), dtype=bool)

    for peak in detected_peaks:
        match_found = False
        for i, true_peak in enumerate(true_peaks):
            if not matched[i] and abs(peak - true_peak) <= tolerance_samples:
                TP += 1
                matched[i] = True
                match_found = True
                break
        if not match_found:
            FP += 1

    FN = len(true_peaks) - TP

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    return {
        'TP': TP, 'FP': FP, 'FN': FN,
        'Sensitivity': round(sensitivity, 3),
        'Precision': round(precision, 3),
        'F1 Score': round(f1, 3)
    }

#################################################################################################################################
#################################################################################################################################

#################################################################################################################################
#########################################>> Plot Functions <<####################################################################


# Function to plot the signals
def plot_signal(signal,fs,text,color):
    t = [i / fs for i in range(len(signal))]
    plt.figure(figsize=(10, 4))
    plt.plot(t[:fs*5], signal[:fs*5], color=color)
    plt.title(text)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (mV)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_filter_response(fs, b, a, title):
    w, h = freqz(b, a, worN=1024, fs=fs)
    z, p, _ = tf2zpk(b, a)
    gd_freq, gd = group_delay((b, a), fs=fs)

    fig, axs = plt.subplots(2, 2, figsize=(10, 4))
    axs[0, 0].plot(w, 20 * np.log10(np.abs(h)))
    axs[0, 0].set_title(f'{title} - Magnitude')
    axs[0, 1].plot(w, np.angle(h))
    axs[0, 1].set_title(f'{title} - Phase')
    axs[1, 0].scatter(np.real(z), np.imag(z), marker='o', label='Zeros')
    axs[1, 0].scatter(np.real(p), np.imag(p), marker='x', label='Poles')
    axs[1, 0].legend(); axs[1, 0].set_title(f'{title} - Pole-Zero Plot')
    axs[1, 1].plot(gd_freq, gd)
    axs[1, 1].set_title(f'{title} - Group Delay')
    plt.tight_layout(); plt.show()


def plot_with_peaks(signal, fs, peaks, threshold):
    t = [i / fs for i in range(len(signal))]
    plt.figure(figsize=(10, 4))

    end = int(fs * 5)

    peaks_in_range = [p for p in peaks if p < end]

    plt.plot(t[:end], signal[:end], label='Integrated Signal', color='green')

    plt.plot([t[p] for p in peaks_in_range],
             [signal[p] for p in peaks_in_range],
             'ro', label='Detected Peaks')

    plt.axhline(y=threshold, color='gray', linestyle='--', label='Threshold')

    plt.title("QRS Detection - Peaks over Threshold (First 5 seconds)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#################################################################################################################################
#################################################################################################################################


#################################################################################################################################
#########################################>> Pan-Tompkins Algorithm <<############################################################

def Pan_Tompkins(path, title):

    ecg_signal, fs = getSignal(path)      # -> call function to get the signal and the frequency

    #First Stage : Bandpass Filter
    lp_filtered = lowpass_filter(ecg_signal)
    plot_signal(lp_filtered, fs, "Lowpass Filter Output", 'blue')

    hp_filtered = highpass_filter(lp_filtered)
    plot_signal(hp_filtered, fs, "Highpass Filter Output (Bandpassed)", 'green')


    plot_filter_response(fs, [1] + [0]*5 + [-2] + [0]*5 + [1], [1, -2, 1], "Lowpass")
    plot_filter_response(fs, np.pad([-1] + [0]*15 + [32] + [0]*15 + [1], (0, 0)), [1, 1], "Highpass")

    # Define LP and HP filter coefficients (same as your code)
    b_lp = [1, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 1]
    a_lp = [1, -2, 1]

    b_hp = np.zeros(33)
    b_hp[0], b_hp[16], b_hp[32] = -1, 32, 1
    a_hp = [1, 1]

    # Combine them (LP followed by HP)
    b_bp = convolve(b_lp, b_hp)
    a_bp = convolve(a_lp, a_hp)

    plot_filter_response(fs, b_bp, a_bp, "Bandpass Filter (LP + HP)")

    #Second Stage : Differentiator
    deriv = derivative(hp_filtered)
    plot_signal(deriv, fs, f"{title} - Derivative", "purple")
    plot_filter_response(fs, np.array([-1, -2, 0, 2, 1]) * (1/8), [1], "Derivative")

    #Third Stage : Squaring
    squared = squaring(deriv)
    plot_signal(squared, fs, f"{title} - Squaring", "green")

    #Fourth Stage : Integration
    window_size = int(0.150 * fs)
    integrated = moving_window_integration(squared, window_size)
    plot_signal(integrated, fs, f"{title} - Integrated", "pink")
    ones = np.ones(window_size) / window_size
    plot_filter_response(fs, ones, [1], "Moving Window Integrator")


    #Fifth Stage : Peak Detection
    peaks, threshold = detect_peaks(integrated, fs)
    plot_with_peaks(integrated, fs, peaks, threshold)

    lms_peaks, threshold_trace = lms_peak_detection(integrated, fs)
    plot_with_peaks(integrated, fs, lms_peaks, threshold_trace[-1])



    #Compare the number of beats detected in the first 5 seconds
    beats = len([p for p in peaks if p < fs * 5])
    print(f"Detected QRS peaks in {title}: {beats} beats in 5 seconds")

    return beats, peaks, lms_peaks, fs



if __name__ == '__main__':
    clean_path = "data/clean_signals/101"
    noisy_path = "data/noisy_signals/108"

    print("----- Clean ECG -----")
    clean_beats, clean_static_peaks, clean_lms_peaks, fs = Pan_Tompkins(clean_path, "Clean ECG")

    print("\n----- Noisy ECG -----")
    noisy_beats, noisy_static_peaks, noisy_lms_peaks, _ = Pan_Tompkins(noisy_path, "Noisy ECG")

    # Load reference annotations
    ann_clean = wfdb.rdann(clean_path, 'atr')
    ann_noisy = wfdb.rdann(noisy_path, 'atr')

    print("\n===== Evaluation: Clean Signal (101) =====")
    print("Static:", evaluate_peaks(clean_static_peaks, ann_clean.sample, fs))
    print("LMS   :", evaluate_peaks(clean_lms_peaks, ann_clean.sample, fs))

    print("\n===== Evaluation: Noisy Signal (108) =====")
    print("Static:", evaluate_peaks(noisy_static_peaks, ann_noisy.sample, fs))
    print("LMS   :", evaluate_peaks(noisy_lms_peaks, ann_noisy.sample, fs))

    print("\n----- Comparison -----")
    print(f"Clean beats: {clean_beats}")
    print(f"Noisy beats: {noisy_beats}")
