import librosa
import numpy as np
import os
import subprocess
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Load audio file
def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)  # y: audio time series, sr: sampling rate
    return y, sr

# Extract MFCC features
def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)  # Averaging across time frames

# Create a dataset from a folder of audio files
def create_dataset(audio_folder):
    features = []
    labels = []
    for label in os.listdir(audio_folder):
        folder_path = os.path.join(audio_folder, label)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith(".wav"):
                    file_path = os.path.join(folder_path, file)
                    y, sr = load_audio(file_path)
                    mfcc_features = extract_features(y, sr)
                    features.append(mfcc_features)
                    labels.append(label)
    return np.array(features), np.array(labels)

# Load your dataset
audio_folder = "/Users/sobithav/Documents/dataset"  # Make sure to provide path to your dataset folder
X, y = create_dataset(audio_folder)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train a classifier (SVM in this case)
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Example of using the trained model for prediction on a new audio file
def predict_scream(file_path):
    y, sr = load_audio(file_path)
    mfcc_features = extract_features(y, sr)
    mfcc_scaled = scaler.transform([mfcc_features])
    prediction = model.predict(mfcc_scaled)
    return prediction[0]

# Predict on a new file and run script if prediction is "screaming"
file_path = "/Users/sobithav/Downloads/3.wav"  # Replace with path to your test audio file
prediction = predict_scream(file_path)
print(f"Prediction for '{file_path}': {prediction}")

if prediction.lower() == "screaming":
    # Path to your SQLite script
    sqlite_script_path = "/Users/sobithav/Library/Application Support/JetBrains/PyCharm2024.3/scratches/sqlite_script.py"  # Replace with the actual path to your script
    try:
        print("Screaming detected. Running SQLite script...")
        subprocess.run(["python", sqlite_script_path], check=True)
        print(f"SQLite script executed successfully for prediction '{prediction}'.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing SQLite script: {e}")
