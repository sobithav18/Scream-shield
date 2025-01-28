import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Directory and class information
data_dir = '/Users/sobithav/Downloads/train_data'
classes = ['Screaming', 'NotScreaming']

# Check if data directory exists
if os.path.exists(data_dir):
    print(f"Directory exists: {data_dir}")
    print("Contents of directory:", os.listdir(data_dir))
else:
    raise FileNotFoundError(f"Directory not found: {data_dir}")

# Load and preprocess data
def load_and_preprocess_data(data_dir, classes, target_shape=(128, 128)):
    data, labels = [], []
    for i, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        for filename in os.listdir(class_dir):
            if filename.endswith('.wav'):
                file_path = os.path.join(class_dir, filename)
                audio_data, sample_rate = librosa.load(file_path, sr=None)
                mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
                mel_spectrogram = tf.image.resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
                data.append(mel_spectrogram.numpy())
                labels.append(i)
    return np.array(data), np.array(labels)

# Data loading and preprocessing
data, labels = load_and_preprocess_data(data_dir, classes)
labels = to_categorical(labels, num_classes=len(classes))  # One-hot encoding
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)

# Visualize a Mel spectrogram
plt.figure(figsize=(10, 4))
plt.imshow(data[0].squeeze(), aspect='auto', cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title(f'Mel Spectrogram (Class: {classes[np.argmax(labels[0])]})')
plt.xlabel('Time')
plt.ylabel('Mel Frequency')
plt.tight_layout()
plt.show()

# Define the model architecture
input_shape = X_train[0].shape
num_labels = len(classes)

# Normalization layer
norm_layer = layers.Normalization()
norm_layer.adapt(X_train)

# Model definition
model = models.Sequential([
    layers.Input(shape=input_shape),
    layers.Resizing(32, 32),  # Downsample
    norm_layer,  # Normalize input
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),  # Regularization
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Regularization
    layers.Dense(num_labels, activation='softmax'),  # Classification layer
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy[1]:.4f}")

# Save the model
model.save('audio_classification_model.keras')

# Load the saved model
model = tf.keras.models.load_model('audio_classification_model.keras')

# Function to test a single audio file
def test_audio(file_path, model, target_shape=(32, 32)):
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
    mel_spectrogram = tf.image.resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
    mel_spectrogram = tf.reshape(mel_spectrogram, (1,) + target_shape + (1,))
    predictions = model.predict(mel_spectrogram)
    class_probabilities = predictions[0]
    
    # Threshold for prioritizing "Screaming"
    screaming_threshold = 0.8
    if class_probabilities[0] > screaming_threshold:
        return class_probabilities, 0
    
    # Return the class with the highest probability
    predicted_class_index = np.argmax(class_probabilities)
    return class_probabilities, predicted_class_index

# Test the model with an audio file
test_audio_file = '/Users/sobithav/Downloads/10.wav'
class_probabilities, predicted_class_index = test_audio(test_audio_file, model)

# Display results
for i, class_label in enumerate(classes):
    print(f"Class: {class_label}, Probability: {class_probabilities[i]:.4f}")
predicted_class = classes[predicted_class_index]
print(f'The audio is classified as: {predicted_class}')
print(f'Confidence: {class_probabilities[predicted_class_index]:.4f}')
