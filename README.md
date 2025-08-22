# Face Detection [AI & ML]

This project provides a complete workflow for creating and using a face recognition system based on a convolutional neural network (CNN) and Haar cascades. It includes scripts for data collection, model training, and real-time face recognition.

## Features

* **`collect_data.py`**: Collects a dataset of 100 face images from your webcam for a specified person.
* **`consolidated_data.py`**: Prepares the collected face images for training by converting them to grayscale, resizing them, and saving them as NumPy arrays and labels using pickle. It also includes an optional step to show a random sample image.
* **`train_model.py`**: Builds and trains a CNN model for face recognition using the prepared dataset. It saves the final model as `final_model.h5` and the label encoder as `label_encoder.pkl` for later use.
* **`recognize.py`**: Utilizes the trained model to perform real-time face recognition on a live webcam feed. It detects faces with a Haar cascade classifier, preprocesses the detected face, and then uses the CNN model to predict the identity, displaying the name on a bounding box.
* **`list_people.py`**: Lists the names of all individuals for whom face data has been collected.
* **`reset.py`**: A utility script to delete all collected face data, saved models, and the label encoder, effectively resetting the project.
* **`.gitignore`**: Specifies files and directories that should be ignored by Git, including `images/`, `data/`, model files (`.h5`), and pickle files (`.pkl`).
* **`haarcascades/haarcascade_frontalface_default.xml`**: The Haar cascade classifier file used for face detection.

---

## Prerequisites

To run this project, you need the following Python libraries installed:

* `opencv-python`
* `numpy`
* `matplotlib`
* `tensorflow`
* `scikit-learn`
* `joblib`

You can install them by running:

```bash
pip install -r requirements.txt
```
## File Structure
The project directory is organized as follows:
```
.
├── Face Detection [AI & ML]/
│   ├── .gitignore
│   ├── consolidated_data.py
│   ├── collect_data.py
│   ├── haarcascades/
│   │   └── haarcascade_frontalface_default.xml
│   ├── list_people.py
│   ├── readme.md
│   ├── recognize.py
│   ├── requirements.txt
│   ├── reset.py
│   └── train_model.py
│   
└── README.md
```

## Usage
Follow these steps to set up and use the face recognition system:

**Collect Face Data**:
Run the collect_data.py script. It will open your webcam and collect 100 face images. You will be prompted to enter a name to label the collected images.

**Consolidate Data for Training**:
Execute consolidated_data.py to preprocess the collected images and prepare them for the training script.

**Train the Model**:
Run train_model.py to train the CNN model on your dataset. This script saves the trained model and the label encoder.

**Perform Face Recognition**:
Once the model is trained, you can use recognize.py to perform real-time face recognition on your webcam.

**List People in the Dataset**:
The list_people.py script can be used at any time to check whose faces are in the dataset.

**Reset the Project**:
To remove all data and models, run reset.py. This is useful if you want to create a new dataset or retrain the model from scratch.