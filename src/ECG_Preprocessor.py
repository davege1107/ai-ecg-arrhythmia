import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from ecgdetectors import Detectors
from scipy.signal import butter, filtfilt

data_root_dir = "./data/physionet.org/files/ecg-arrhythmia/one/WFDBRecords"

# The preprocess_signal function filters and 
# downsamplesthe raw ECG signal to remove noise
# and reduce computational complexity.
def preprocess_signal(signal):

    b, a = butter(4, 40 / (500 / 10), btype='low')
    filtered_signal = filtfilt(b, a, signal, axis=-1)

    downsampled_signal = filtered_signal[:, ::10] 

    return downsampled_signal

# visualizes the processed ECG signal using Matplotlib,
# showing amplitude against time.
def plot_ecg(signal, sampling_rate, title="ECG Signal"):
    time = np.arange(0, len(signal)) / sampling_rate
    plt.figure(figsize=(12, 6))
    plt.plot(time, signal)
    plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

def print_file_info(file, r_peaks, rr_intervals):
    print(f"Insights for ECG signal {file}:")
    print(f"Number of QRS complexes detected: {len(r_peaks)}")
    print(f"Average RR interval: {np.mean(rr_intervals)} seconds")
    print(f"Minimum RR interval: {np.min(rr_intervals)} seconds")
    print(f"Maximum RR interval: {np.max(rr_intervals)} seconds")
    print("------------------------------------------------------")


def process_all_ecg_files(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".mat"):
                file_path = os.path.join(root, file)
                try:
                    mat_data = scipy.io.loadmat(file_path)

                    # Assuming shape is (12, n_samples)
                    ecg_data = mat_data['val']

                    ecg_data = preprocess_signal(ecg_data)

                    # Lead II is usually index 1
                    lead_II = ecg_data[1]  # array of shape (n_samples,)
                    sampling_rate = 500  # Hz, adjust if known differently

                    # plots ecg data for each mat file. Feel free to comment out to speed
                    # up the processing
                    plot_ecg(ecg_data[0], sampling_rate, title = file)

                    detectors = Detectors(sampling_frequency=sampling_rate)

                    # Detect R-peaks using Pan-Tompkins
                    r_peaks = detectors.pan_tompkins_detector(lead_II)

                    # Compute RR intervals (in seconds)
                    rr_intervals_sec = np.diff(r_peaks) / sampling_rate

                    print_file_info(file, r_peaks, rr_intervals_sec)

                except Exception as e:
                    print(f"Error processing file {file}: {e}")


if __name__ == "__main__":
    process_all_ecg_files(data_root_dir)