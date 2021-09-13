import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
from IPython import display
import UploadData

class MalGAN():
    def build_blackbox_detector(self):

        if self.blackbox is 'MLP':
            blackbox_detector = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
                                              solver='sgd', verbose=0, tol=1e-4, random_state=1,
                                              learning_rate_init=.1)
        return blackbox_detector


    def build_generator():
        model = tf.keras.Sequential()
        model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((7, 7, 256)))
        assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 7, 7, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 14, 14, 64)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 28, 28, 1)

        return model

    def build_substitute_detector():
        return Sequential([
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_h, img_w, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),#the number of Parmeter:size*size*dim_of_input*number+bias(number)
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=2),
        layers.Flatten(),#if it is not flattened, only the last layer can be fed into the dense
        layers.Dense(1024, activation='relu'),
        layers.Dense(2, activation='relu'),#there are two typs of output, malware and benign, so the dense has 2 neurons
        ])

    def __init__(self):

        IMAGE_WIDTH = 256
        IMAGE_HEIGHT = 256
        # Build and Train blackbox_detector
        self.blackbox_detector = self.build_blackbox_detector()

        # Build and compile the substitute_detector
        self.substitute_detector = self.build_substitute_detector()
        self.substitute_detector.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()
        example = Input(shape=(self.IMAGE_WIDTH,self.IMAGE_HEIGHT))
        noise = Input(shape=(self.IMAGE_WIDTH,IMAGE_HEIGHT))
        input = [example, noise]#要改
        malware_examples = self.generator(input)

        # For the combined model we will only train the generator
        self.substitute_detector.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.substitute_detector(malware_examples)

        # The combined model  (stacked generator and substitute_detector)
        # Trains the generator to fool the discriminator
        self.combined = Model(input, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

        self.train_ds,self.train_mal_ds,self.train_ben_ds= UploadData.get_train_dataset()#get the dataset
        self.vali_ds,self.vali_mal_ds,self.vali_ben_ds= UploadData.get_vali_dataset()

        self.trian_x, self.train_y = next(iter(train_ds))
        self.train_mal_x, self.train_mal_y= next(iter(train_mal_ds))
        self.train_ben_x, self.train_ben_y= next(iter(train_ben_ds))

        self.vali_x, self.vali_y = next(iter(vali_ds))
        self.vali_mal_x, self.vali_mal_y= next(iter(vali_mal_ds))
        self.vali_ben_x, self.vali_ben_y= next(iter(vali_ben_ds))
        

    def train(self, epochs, batch_size=32):
        #load the dataset
        # Train blackbox_detctor
        self.blackbox_detector.fit(Concatenate(axis=1)([self.trian_x,  self.vali_x]),
                                    Concatenate(axis=1)([self.trian_y,  self.vali_y]))
  
        ytrain_ben_blackbox = self.blackbox_detector.predict(train_ben_x)
        Original_Train_TRR = self.blackbox_detector.score(train_mal_x,train_mal_y)
        Original_Test_TRR = self.blackbox_detector.score(test_mal_x, test_mal_y)
        Train_TRR, Test_TRR = [], []

        for epoch in range(epochs):
    
            for step in range(1):#range(xtrain_mal.shape[0] // batch_size):
                # ---------------------
                #  Train substitute_detector
                # ---------------------

                # Select a random batch of malware examples
                idx = tf.random.randint(0, train_mal_x.shape[0], batch_size)
                xmal_batch = xtrain_mal[idx]
              
                noise = tf.random.normal([BATCH_SIZE, noise_dim])
                idx = tf.random.randint(0, xmal_batch.shape[0], batch_size)
                xben_batch = xtrain_ben[idx]
                yben_batch = ytrain_ben_blackbox[idx]

                gen_examples = self.generator.predict([xmal_batch, noise])
                ymal_batch = self.blackbox_detector.predict(np.ones(gen_examples.shape)*(gen_examples > 0.5))