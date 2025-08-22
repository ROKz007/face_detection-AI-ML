import pickle
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ---------------------------
# Load dataset
# ---------------------------
data_dir = os.path.join(os.getcwd(), 'data')

with open(os.path.join(data_dir, "images.p"), 'rb') as f:
    images = pickle.load(f)

with open(os.path.join(data_dir, "labels.p"), 'rb') as f:
    labels = pickle.load(f)

# Normalize images
images = images / 255.0
images = images.reshape(images.shape[0], 100, 100, 1)  # CNN input

# Encode labels (text -> numbers)
encoder = LabelEncoder()
labels_encoded = encoder.fit_transform(labels)
labels_encoded = to_categorical(labels_encoded)

# ---------------------------
# Build CNN model
# ---------------------------
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(labels_encoded.shape[1], activation='softmax')  # output layer
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ---------------------------
# Train model
# ---------------------------
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint("best_model.h5", save_best_only=True)
]

history = model.fit(
    images, labels_encoded,
    epochs=15,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks
)

# ---------------------------
# Save final model
# ---------------------------
model.save("final_model.h5")
print("✅ Training complete. Model saved as final_model.h5")

# ---------------------------
# Save label encoder (so recognize.py knows class names)
# ---------------------------
import joblib
joblib.dump(encoder, "label_encoder.pkl")
print("✅ Label encoder saved as label_encoder.pkl")
