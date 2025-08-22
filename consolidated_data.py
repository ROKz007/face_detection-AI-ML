import cv2
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import random

# Directories
data_dir = os.path.join(os.getcwd(), 'data')
img_dir = os.path.join(os.getcwd(), 'images')

# Create data folder if it doesn't exist
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

image_data = []
labels = []

# Read images and extract labels
for i in os.listdir(img_dir):
    image = cv2.imread(os.path.join(img_dir, i))
    image = cv2.resize(image, (100, 100))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_data.append(image)
    labels.append(str(i).split("_")[0])

# Convert to numpy arrays
image_data = np.array(image_data)
labels = np.array(labels)

# Show a random sample instead of fixed index
idx = random.randint(0, len(image_data)-1)
plt.imshow(image_data[idx], cmap="gray")
plt.title(f"Sample: {labels[idx]}")
plt.show()

# Save data
with open(os.path.join(data_dir, "images.p"), 'wb') as f:
    pickle.dump(image_data, f)

with open(os.path.join(data_dir, "labels.p"), 'wb') as f:
    pickle.dump(labels, f)

print(f"Dataset saved ✅ | Images: {image_data.shape}, Labels: {labels.shape}")
