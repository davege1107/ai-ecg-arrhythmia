import os
import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from ECG_Preprocessor import preprocess_signal  # your provided function

def extract_features(signal, sampling_rate=500):
    # Extract meaningful features
    lead_II = signal[1]
    rr_intervals = np.diff(np.where(lead_II > np.mean(lead_II))[0]) / sampling_rate
    return np.array([
        np.mean(rr_intervals),
        np.std(rr_intervals),
        np.min(rr_intervals),
        np.max(rr_intervals),
    ])

def parse_label(hea_file_path):
    with open(hea_file_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('#Dx'):
            label = line.strip().split(' ')[1].split(',')[0]  # crude extraction, might need refining
            return label
    return None

def build_dataset(data_dir):
    X = []
    y = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.mat'):
                mat_path = os.path.join(root, file)
                hea_path = mat_path.replace('.mat', '.hea')
                print(f"Processing {mat_path} and {hea_path}")
                mat_data = scipy.io.loadmat(mat_path)
                signal = mat_data['val']

                signal = preprocess_signal(signal)
                features = extract_features(signal)

                label = parse_label(hea_path)

                if label is not None:
                    X.append(features)
                    y.append(label)
    return np.array(X), np.array(y)

# Main script
data_root = "./data/physionet.org/files/ecg-arrhythmia/one/WFDBRecords"
X, y = build_dataset(data_root)

# Encode labels if necessary
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train
clf = GaussianNB()
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
