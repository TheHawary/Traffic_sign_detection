# Traffic Sign Recognition System 🚦

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow%202.x-orange)
![OpenCV](https://img.shields.io/badge/Vision-OpenCV-green)
![Status](https://img.shields.io/badge/Status-Completed-success)




## Project Overview
This project implements a Convolutional Neural Network (CNN) to classify 5998 images across **58 distinct classes** of traffic signs. 
It is designed to assist autonomous vehicle systems in recognizing road regulations in real-time.

The model takes raw images, preprocesses them to a standardized format, and predicts the traffic sign class using a deep learning architecture trained on custom traffic data.

## Technical Approach

## Dataset
You can find the dataset used in training this model on (https://www.kaggle.com/datasets/dmitryyemelyanov/chinese-traffic-signs) from Kaggle.

**To reproduce results:**
1. Download the dataset from the link above.
2. Extract the files into a folder named `traffic_Data` in the root directory.
3. Ensure the structure is: `traffic_Data/DATA/[class_folders]`.

### 1. Data Preprocessing
* **Input Resolution:** All images are resized to **32x32 pixels** (RGB).
* **Normalization:** Pixel intensity values are scaled to the range `[0, 1]` (divided by 255.0).
* **Encoding:** Labels are one-hot encoded for multi-class classification (58 classes).
* **Splitting:** The dataset is split into **80% Training** and **20% Validation**.

### 2. CNN Architecture
The model utilizes a sequential architecture optimized for feature extraction from low-resolution images:

| Layer Type | Configuration | Output Shape |
| :--- | :--- | :--- |
| **Input** | RGB Image | (32, 32, 3) |
| **Conv2D** | 32 filters, 3x3 kernel, ReLU | Feature Map |
| **MaxPooling** | 2x2 pool size | Downsampled Map |
| **Conv2D** | 64 filters, 3x3 kernel, ReLU | Feature Map |
| **MaxPooling** | 2x2 pool size | Downsampled Map |
| **Conv2D** | 64 filters, 3x3 kernel, ReLU | Feature Map |
| **Flatten** | - | 1D Vector |
| **Dense** | 64 neurons, ReLU | Fully Connected |
| **Output** | 58 neurons, Softmax | Class Probabilities |

### 3. Training Configuration
* **Optimizer:** Adam
* **Loss Function:** Categorical Crossentropy
* **Epochs:** 200
* **Batch Size:** 16

## Performance
* **Training Accuracy:** [e.g., 99.1%]
* **Validation Accuracy:** [e.g., 96.5%]



## Dependencies
To run this project, install the following libraries:

```txt
tensorflow
opencv-python
numpy
pandas
matplotlib
scikit-learn
