import wfdb
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, freqz, tf2zpk, group_delay, lfilter
import numpy as np
from scipy.signal import find_peaks

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

def detect_peaks(signal,fs):
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    threshold = mean_val + 0.5 * std_val

    min_distance = int(0.2 * fs)

    peaks, _ = find_peaks(signal, height=threshold, distance=min_distance)

    return peaks, threshold

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

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
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
    plt.figure(figsize=(12, 5))

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



    #Second Stage : Differentiator
    deriv = derivative(hp_filtered)
    plot_signal(deriv, fs, f"{title} - Derivative", "purple")
    plot_filter_response(fs, np.array([-1, -2, 0, 2, 1]) * (1/8), [1], "Derivative")

    #Third Stage : Squaring
    squared = squaring(deriv)
    plot_signal(squared, fs, f"{title} - Squaring", "yellow")

    #Fourth Stage : Integration
    window_size = int(0.150 * fs)
    integrated = moving_window_integration(squared, window_size)
    plot_signal(integrated, fs, f"{title} - Integrated", "pink")
    ones = np.ones(window_size) / window_size
    plot_filter_response(fs, ones, [1], "Moving Window Integrator")


    #Fifth Stage : Peak Detection
    peaks, threshold = detect_peaks(integrated, fs)
    plot_with_peaks(integrated, fs, peaks, threshold)

    #Compare the number of beats detected in the first 5 seconds
    beats = len([p for p in peaks if p < fs * 5])
    print(f"Detected QRS peaks in {title}: {beats} beats in 5 seconds")

    return beats



if __name__ == '__main__':


        clean_path = "data/clean_signals/101"   # -> load clean data path
        noisy_path = "data/noisy_signals/108"   # -> load noisy data path

        print("----- Clean ECG -----")
        clean_beats = Pan_Tompkins(clean_path, "Clean ECG")

        print("\n----- Noisy ECG -----")
        noisy_beats = Pan_Tompkins(noisy_path, "Noisy ECG")


        print("\n----- Comparison -----")
        print(f"Clean beats: {clean_beats}")
        print(f"Noisy beats: {noisy_beats}")