import os
import glob
import json
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Masking, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import tensorflow as tf
from tqdm import tqdm

# Configure GPU memory growth to avoid taking all GPU memory at once
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print("GPU is available:")
    for device in physical_devices:
        print(f"  {device}")
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except RuntimeError as e:
            print(f"Error setting memory growth: {e}")
else:
    print("No GPU devices found. Running on CPU.")

def load_data(data_dir):
    """
    Loads JSON files from the given directory and converts each into a sequence
    of feature vectors. Each feature vector is built by concatenating the 21
    hand landmarks for Left and Right hands (if available; otherwise, zeros are used).
    """
    data = []    # list of sequences; each sequence is a numpy array of shape (T, feature_dim)
    labels = []  # list of string labels corresponding to each sequence
    
    # Find all JSON files in the folder
    file_paths = glob.glob(os.path.join(data_dir, '*.json'))
    if not file_paths:
        print("No JSON files found in", data_dir)
        return data, labels

    for file_path in file_paths:
        with open(file_path, 'r') as f:
            sample = json.load(f)
        
        # The label for the sample (e.g. the signed word)
        label = sample.get("word")
        
        # Get the list of frame data and sort it by frame number (if necessary)
        frames = sample.get("frameData", [])
        frames = sorted(frames, key=lambda x: x.get("frame", 0))
        
        sample_sequence = []
        for frame in frames:
            # Initialize features for left and right hands (21 landmarks × 3 coordinates each)
            left_features = np.zeros(21 * 3, dtype=np.float32)
            right_features = np.zeros(21 * 3, dtype=np.float32)
            
            # Process each detected hand in the frame
            for hand in frame.get("hands", []):
                landmarks = hand.get("landmarks", [])
                # Flatten the landmarks into a 1D array: [x1, y1, z1, x2, y2, z2, ..., x21, y21, z21]
                flattened = np.array([[lm["x"], lm["y"], lm["z"]] for lm in landmarks], dtype=np.float32).flatten()
                
                # Depending on the handedness, assign the flattened landmarks
                if hand.get("handedness", "") == "Left":
                    left_features = flattened
                elif hand.get("handedness", "") == "Right":
                    right_features = flattened
                    
            # Concatenate left and right hand features into one feature vector (length = 126)
            frame_feature = np.concatenate([left_features, right_features])
            sample_sequence.append(frame_feature)
        
        # Append the sequence (converted to a numpy array) and its label
        data.append(np.array(sample_sequence, dtype=np.float32))
        labels.append(label)
    
    return data, labels

def main():
    # Folder where JSON files are stored
    data_dir = 'parsed-output'
    
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    # Load data and labels from JSON files
    data, labels = load_data(data_dir)
    if not data:
        print("No data loaded. Exiting.")
        return

    print(f"Loaded {len(data)} samples.")

    # Encode string labels into integers (e.g., "kaos" -> 0, "hello" -> 1, etc.)
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    
    # Save the label encoder for consistent label mapping
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Determine the maximum sequence length (number of frames) to pad all sequences equally
    max_seq_length = max(seq.shape[0] for seq in data)
    print("Max sequence length:", max_seq_length)
    
    # Pad sequences with zeros (padding is done at the end of each sequence)
    # After padding, data_padded will be a numpy array of shape (num_samples, max_seq_length, feature_dim)
    data_padded = pad_sequences(data, maxlen=80, dtype='float32', padding='post', truncating='post')

    print("Data:", data_padded)
    print("Labels:", labels_encoded)
    
    # Split data into training and test sets
    # X_train, X_test, y_train, y_test = train_test_split(
    #     data_padded, labels_encoded, test_size=0.2, random_state=42
    # )
    
    X_train = data_padded
    y_train = labels_encoded
    X_test = data_padded
    y_test = labels_encoded

    # Build a simple LSTM model for sequence classification
    num_classes = len(label_encoder.classes_)
    feature_dim = data_padded.shape[2]  # Should be 126 (i.e. 21 landmarks * 3 coordinates * 2 hands)
    
    # Build LSTM model with explicit input shape
    model = Sequential([
        LSTM(64, input_shape=(80, feature_dim), return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=True),
        Dropout(0.2),
        LSTM(16),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    
    # Use a lower learning rate for better stability
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    # Print model summary to verify parameters
    model.summary()
    
    # Train the model
    epochs = 500
    batch_size = 16
    
    early_stopping = EarlyStopping(monitor='accuracy', patience=30)

    history = model.fit(X_train, y_train,
                   validation_data=(X_test, y_test),
                   epochs=epochs,
                   batch_size=batch_size,
                   callbacks=[early_stopping])
    
    print("Evaluating model on test data...")
    loss, accuracy = model.evaluate(X_test, y_test)
    
    # Save the trained model for future use
    model.save('sign_language_model.keras')
    
    print("Model and label encoder saved.")
    print(f"Test accuracy: {accuracy:.4f}")

    test_model()

def test_model():
    # Load the trained model and label encoder
    model = tf.keras.models.load_model('sign_language_model.keras')
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Load test data
    data_dir = 'test-data'
    data, labels = load_data(data_dir)
    if not data:
        print("No data loaded. Exiting.")
        return
    
    # Encode string labels into integers using the loaded label encoder
    labels_encoded = label_encoder.transform(labels)
    
    # Pad sequences
    data_padded = pad_sequences(data, 
                              maxlen=80, 
                              dtype='float32',
                              padding='post',  # Keep post-padding
                              truncating='post',
                              value=0.0)  # Explicit padding value
    
    # Evaluate the model on the test data
    loss, accuracy = model.evaluate(data_padded, labels_encoded)
    print(f"Test accuracy: {accuracy:.4f}")

    print("Predictions:")
    predictions = model.predict(data_padded)
    predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))
    for i, label in enumerate(predicted_labels):
        print(f"Predicted: {label}, True: {labels[i]}")

if __name__ == '__main__':
    main()