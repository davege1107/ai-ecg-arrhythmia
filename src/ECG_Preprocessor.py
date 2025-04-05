import os
import scipy.io
import numpy as np
from scipy.signal import butter, filtfilt
import biosppy.signals.ecg as ecg
import matplotlib.pyplot as plt

# This is taken from https://medium.com/@shahbaz.gondal588/understanding-ecg-signal-processing-with-python-b9dd4ea68682

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

# The detect_qrs_complex function uses BioSPPy to detect QRS
# complexes in the ECG signal, identifying heartbeats.
def detect_qrs_complex(signal, sampling_rate):

    qrs_indices = ecg.ecg(signal=signal, sampling_rate=sampling_rate, show=False)['rpeaks']
    return qrs_indices

# The calculate_rr_intervals function computes RR intervals
# (the time between successive heartbeats) from the detected QRS complexes.
def calculate_rr_intervals(qrs_indices, sampling_rate):

    rr_intervals = np.diff(qrs_indices) / sampling_rate
    return rr_intervals




data_root_dir = "./data/physionet.org/files/ecg-arrhythmia/one/WFDBRecords"



# The process_all_ecg_files function iterates through all ECG files
# in a specified directory, processes each file using the
# aforementioned functions, and prints insights about each ECG signal.
def process_all_ecg_files(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".mat"):
                file_path = os.path.join(root, file)
                try:
                    mat_data = scipy.io.loadmat(file_path)
                    ecg_signal = mat_data['val']
                    processed_signal = preprocess_signal(ecg_signal)
                    plot_ecg(processed_signal[0], sampling_rate=500, title=file)
                    qrs_indices = detect_qrs_complex(processed_signal[0], sampling_rate=500)
                    rr_intervals = calculate_rr_intervals(qrs_indices, sampling_rate=500)
                    print(f"Insights for ECG signal {file}:")
                    print(f"Number of QRS complexes detected: {len(qrs_indices)}")
                    print(f"Average RR interval: {np.mean(rr_intervals)} seconds")
                    print(f"Minimum RR interval: {np.min(rr_intervals)} seconds")
                    print(f"Maximum RR interval: {np.max(rr_intervals)} seconds")
                    print("------------------------------------------------------")
                except Exception as e:
                    print(f"Error processing file {file}: {e}")


if __name__ == "__main__":
    process_all_ecg_files(data_root_dir)