#Importing required libraries
import os
import json
import tempfile
import requests
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds#Loading MNIST train and test dataset
#as_supervised=True, will return tuple instead of a dictionary for image and label
(ds_train, ds_test), ds_info = tfds.load("mnist", split=['train','test'], with_info=True, as_supervised=True)#to select the 'image' and 'label' using indexing coverting train and test dataset to a numpy array
array = np.vstack(tfds.as_numpy(ds_train))
X_train = np.array(list(map(lambda x: x[0], array)))
y_train = np.array(list(map(lambda x: x[1], array)))
X_test = np.array(list(map(lambda x: x[0], array)))
y_test = np.array(list(map(lambda x: x[1], array)))#setting batch_size and epochs
epoch=10
batch_size=128#Creating input data pipeline for train and test dataset
# Function to normalize the images
def normalize_image(image, label):
  #Normalizes images from uint8` to float32
  return tf.cast(image, tf.float32) / 255., label# Input data pipeline for test dataset
#Normalize the image using map function then cache and shuffle the #train dataset 
# Create a batch of the training dataset and then prefecth for #overlapiing image preprocessing(producer) and model execution work #(consumer)ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(batch_size)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)# Input data pipeline for test dataset (No need to shuffle the test #dataset)
ds_test = ds_test.map(
    normalize_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(batch_size)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)# Build the model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
  tf.keras.layers.Dense(128,activation='relu'),
  tf.keras.layers.Dense(196, activation='softmax')
])#Compile the model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy'],)#Fit the model
model.fit(
    ds_train,
    epochs=epoch,
    validation_data=ds_test,
    verbose=2)

MODEL_DIR='tf_model'
version = "1"
export_path = os.path.join(MODEL_DIR, str(version))#Save the model 
model.save(export_path, save_format="tf")
print('\nexport_path = {}'.format(export_path))
!dir {export_path}