import tensorflow as tf;
from tensorflow import keras;

from tensorflow.keras.models import *
import tensorflow.keras.layers as layers
import tensorflow.keras.applications as applications
from tensorflow.keras.layers  import *
import numpy as np
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from CNN_models import Malware_detection_model

tf.__version__



TRAINING_DATA_DIRECTORY = '/content/dataset/Train'
TEST_DATA_DIRECTORY = '/content/dataset/Test'
COLOR_MODE = "greyscale"
BATCH_SIZE = 128
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
SEED = 1337

M_par = Malware_detection_model(IMAGE_WIDTH,IMAGE_HEIGHT,"M1")
model = M_par.Model
model.summary()

training_set = keras.preprocessing.image_dataset_from_directory(
    TRAINING_DATA_DIRECTORY,
    labels="inferred",
    label_mode="int",
    #color_mode=COLOR_MODE
    validation_split=0.15,
    subset='training',
    seed=SEED,
    batch_size = BATCH_SIZE,   
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH), 
    #interpolation = "area"
    shuffle=True,
)



validation_set = keras.preprocessing.image_dataset_from_directory(
    TRAINING_DATA_DIRECTORY,
    labels="inferred",
    label_mode="int",
    validation_split=0.15,
    subset='validation',
    seed=SEED,
    batch_size = BATCH_SIZE,  
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH), 
)



CLASS_NAMES = training_set.class_names
print(CLASS_NAMES)




AUTOTUNE = tf.data.AUTOTUNE

training_set = training_set.cache().shuffle(11929).prefetch(buffer_size=AUTOTUNE)
validation_set = validation_set.cache().prefetch(buffer_size=AUTOTUNE)





model_optimizer = optimizers.Adam(learning_rate= M_par.rate)
loss_function = tf.keras.losses.BinaryCrossentropy()
model_metrics = [tf.keras.metrics.Accuracy(name="accuracy", dtype=None),tf.keras.metrics.Precision( thresholds=None, top_k=None, class_id=None, name=None, dtype=None), tf.keras.metrics.Recall( thresholds=None, top_k=None, class_id=None, name=None, dtype=None)]




model.compile(optimizer = model_optimizer, loss = loss_function,
              metrics = model_metrics)




model_history = model.fit(training_set, epochs = M_par.epochs, validation_data = validation_set)


