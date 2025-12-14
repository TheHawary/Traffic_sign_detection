import cv2                                   # OpenCV
import numpy as np                           # Numpy
import tensorflow as tf                      # TensorFlow
import pandas as pd                          # Pandas 
import os                                    # Operating system
from matplotlib import pyplot as plt         # Matplot Libarary


# Function to load the dataset
def load_dataset(dataset_path):
    images = []
    labels = []
    
    # Loop through each class in the dataset
    for class_index in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_index)
        if os.path.isdir(class_path):
            # Loop through each image in the class directory
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)

                # Read the image and resize it to a fixed size of 32x32
                image = cv2.imread(image_path)
                image = cv2.resize(image, (32, 32))
                images.append(image)
                labels.append(int(class_index))  # Using directory name as labels
                
    return np.array(images), np.array(labels)



# Load the dataset
dataset_path = "traffic_Data/DATA"
images, labels = load_dataset(dataset_path)

# Normalize pixel values to the range [0, 1]
images = images / 255.0

# Spliting the dataset into training and validation sets 
from sklearn.model_selection import train_test_split
train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Convert labels to one-hot encoding
from keras.utils import to_categorical
num_classes = 58
train_labels = to_categorical(train_labels, num_classes)
val_labels = to_categorical(val_labels, num_classes)


# Adjusting the model layers

model = tf.keras.Sequential([
    # Add a 2D convolution layer with 32 filters, each of size 3x3
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)), 

    # Add a max-pooling layer with a pool size of 2x2     
    tf.keras.layers.MaxPooling2D((2, 2)),                    
    
    # Add another convolution layer with 64 filters of size 3x3
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),         
   
    # Add another max-pooling layer with a pool size of 2x2
    tf.keras.layers.MaxPooling2D((2, 2)),                
    
    # Add another convolution layer with 64 filters of size 3x3
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),   
    
    # Flatten the 3D output of the previous layer into a 1D tensor
    tf.keras.layers.Flatten(),                 
    
    # Add a fully connected layer with 64 units and ReLU activation                                          
    tf.keras.layers.Dense(64, activation='relu'),           
   
    # Add the output layer with 58 units and softmax activation                             
    tf.keras.layers.Dense(58, activation='softmax')                                      
])


# Compiling the model

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Training the model
history = model.fit(train_images, train_labels, epochs=200, batch_size=16, validation_data=(val_images, val_labels))


# Saving the model
model.save("trained_model", include_optimizer=True)