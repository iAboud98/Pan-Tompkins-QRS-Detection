import wfdb
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import numpy as np
from scipy.signal import find_peaks


def getSignal(path):
    # تحميل الإشارة من الملف
    record = wfdb.rdrecord(path)
    annotation = wfdb.rdann(path, 'atr')

    # first chanel in the signal
    ecg_signal = record.p_signal[:, 0]
    fs = record.fs
    print(f"Sampling Frequency (fs): {fs} Hz")

    # محور الزمن
    t = [i / fs for i in range(len(ecg_signal))]

    # ploting the first 10 seconds
    plt.plot(t[:fs * 10], ecg_signal[:fs * 10])
    plt.title("ECG Signal - First 10 Seconds")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (mV)")
    plt.grid(True)
    plt.show()

    return ecg_signal, fs

# min  max is the bandwidth limits
def bandpassFilter(signal ,min ,max,fs,order):
    # we do this bc when we use the butter() it doesnt deal with hz it deals with the nyquest domain
    nq=fs/2  # this is the highest frequency  of the signal

    lowcut=min/nq
    highcut=max/nq

    b, a = butter(order, [lowcut, highcut], btype='band')# this returns
    filtered_signal = filtfilt(b, a, signal)

    return filtered_signal


def derivative(signal, fs):
    #هاي بتروح بتعملي مصفوفة حجمها زي حجم السغنال
    diff = np.zeros_like(signal)
    T=1/fs

    for i in range(2, len(signal) - 2):
        diff[i] = (1 / (8 * T)) * (-signal[i - 2] - 2 * signal[i - 1] + 2 * signal[i + 1] + signal[i + 2])

    return diff


def squaring(signal):
    squared = signal * signal
    return squared


#هاد ما فهمته
def moving_window_integration(signal, window_size):
    # نعمل نافذة ثابتة القيم (box window) بطول معيّن
    window = np.ones(window_size) / window_size
    # نطبّق الفلتر التراكمي (convolution) بين الإشارة والنافذة
    integrated_signal = np.convolve(signal, window, mode='same')
    return integrated_signal


#هاد كمتن

def detect_peaks(signal,fs):
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    threshold = mean_val + 0.5 * std_val

    # نكتشف القمم فوق العتبة، نستخدم distance كـ فترة راحة بين الضربات (مثلاً 200ms)
    min_distance = int(0.2 * fs)

    peaks, _ = find_peaks(signal, height=threshold, distance=min_distance)

    return peaks, threshold


def plot_signal(signal,fs,text,color):
    t = [i / fs for i in range(len(signal))]
    plt.figure(figsize=(10, 4))
    plt.plot(t[:fs*10], signal[:fs*10], color=color)
    plt.title(text)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (mV)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_with_peaks(signal, fs, peaks, threshold):
    t = [i / fs for i in range(len(signal))]
    plt.figure(figsize=(12, 5))

    # نحدد نهاية أول 10 ثواني
    end = int(fs * 10)

    # فقط القمم الموجودة خلال أول 10 ثواني
    peaks_in_range = [p for p in peaks if p < end]

    # رسم الإشارة فقط لأول 10 ثواني
    plt.plot(t[:end], signal[:end], label='Integrated Signal', color='green')

    # رسم القمم فوق الإشارة
    plt.plot([t[p] for p in peaks_in_range],
             [signal[p] for p in peaks_in_range],
             'ro', label='Detected Peaks')

    # رسم خط العتبة
    plt.axhline(y=threshold, color='gray', linestyle='--', label='Threshold')

    plt.title("QRS Detection - Peaks over Threshold (First 10 seconds)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def Pan_Tompkins(path, title, color):
    # قراءة الإشارة
    ecg_signal, fs = getSignal(path)

    # الفلترة
    filtered = bandpassFilter(ecg_signal, 5, 15, fs, 4)
    plot_signal(filtered, fs, f"{title} - Filtered", color)

    # الاشتقاق
    deriv = derivative(filtered, fs)
    plot_signal(deriv, fs, f"{title} - Derivative", "purple")

    # التربيع
    squared = squaring(deriv)
    plot_signal(squared, fs, f"{title} - Squaring", "yellow")

    # الدمج
    window_size = int(0.150 * fs)
    integrated = moving_window_integration(squared, window_size)
    plot_signal(integrated, fs, f"{title} - Integrated", "pink")

    # الكشف
    peaks, threshold = detect_peaks(integrated, fs)
    plot_with_peaks(integrated, fs, peaks, threshold)

    # طباعة عدد القمم
    beats = len([p for p in peaks if p < fs * 10])
    print(f"Detected QRS peaks in {title}: {beats} beats in 10 seconds")

    return beats



# Press the green button in the gutter to run the script.
if __name__ == '__main__':


        clean_path = "data/clean_signals/101"
        noisy_path = "data/noisy_signals/108"

        print("----- Clean ECG -----")
        clean_beats = Pan_Tompkins(clean_path, "Clean ECG", "green")

        print("\n----- Noisy ECG -----")
        noisy_beats = Pan_Tompkins(noisy_path, "Noisy ECG", "red")


        print("\n----- Comparison -----")
        print(f"Clean beats: {clean_beats}")
        print(f"Noisy beats: {noisy_beats}")