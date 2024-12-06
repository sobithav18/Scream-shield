import librosa
import numpy as np
import sounddevice as sd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from scipy.io.wavfile import write

# Load audio file
def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)  # y: audio time series, sr: sampling rate
    return y, sr

# Extract MFCC features
def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    # Optionally include delta MFCCs or other features
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
audio_folder = "/Users/sobithav/Documents/dataset"  # Provide path to your dataset folder
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

# Predict on recorded audio
def predict_scream_audio(audio_data, sr):
    try:
        mfcc_features = extract_features(audio_data, sr)
        mfcc_scaled = scaler.transform([mfcc_features])
        prediction = model.predict(mfcc_scaled)
        return "Screaming" if prediction[0] == 'screaming' else "Not Screaming"
    except Exception as e:
        return f"Error processing audio: {e}"

# Record live audio and predict
def record_and_predict():
    print("Recording... Speak now!")
    duration = 5  # Record for 5 seconds
    sr = 22050  # Sample rate
    audio_data = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    print("Recording complete. Analyzing...")
    audio_data = audio_data.flatten()  # Flatten the array (from stereo to mono)
    prediction = predict_scream_audio(audio_data, sr)
    print(f"Prediction: {prediction}")

# Main interactive loop
while True:
    choice = input("Type 'record' to record your voice or 'exit' to quit: ").strip().lower()
    if choice == 'record':
        record_and_predict()
    elif choice == 'exit':
        print("Exiting the program.")
        break
    else:
        print("Invalid input. Please type 'record' or 'exit'.")
