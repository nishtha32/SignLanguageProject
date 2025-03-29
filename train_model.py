import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

def load_dataset(dataset_dir):
    data = []
    labels = []

    for file in os.listdir(dataset_dir):
        if file.endswith('.csv'):
            file_path = os.path.join(dataset_dir, file)
            df = pd.read_csv(file_path)
            data.append(df.values)
            labels.extend([file.split('_')[0]] * len(df))  # Use the filename as the label

    data = np.vstack(data)  # Combine all CSV data
    return data, np.array(labels)

def preprocess_data(data, labels):
    # Map labels to integers
    label_map = {label: idx for idx, label in enumerate(np.unique(labels))}
    labels = np.array([label_map[label] for label in labels])

    # Normalize data
    data = data / np.max(data)

    # Convert labels to one-hot encoding
    labels = to_categorical(labels, num_classes=len(label_map))
    return data, labels, label_map

def build_model(input_shape, num_classes):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    dataset_dir = "dataset"
    data, labels = load_dataset(dataset_dir)
    data, labels, label_map = preprocess_data(data, labels)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Train the model
    model = build_model(X_train.shape[1], len(label_map))
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15, batch_size=32)

    # Save the model and label map
    model.save("models/sign_language_model.h5")
    with open("models/label_map.npy", "wb") as f:
        np.save(f, label_map)
