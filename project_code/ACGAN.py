# from Models import MalwareDetectionModels
# from UploadData import UploadData
import tensorflow as tf
import numpy as np
import os
import tensorflow.keras as keras
from tensorflow.keras import layers, models, applications, optimizers, losses, metrics
from keras.layers.merge import Concatenate
from keras import layers
from keras.layers import Input
from keras.models import Model
from load_data import UploadData


class MasterGAN:
    BUFFER_SIZE = 60000
    BATCH_SIZE = 256

    def __init__(self, blackbox='M11', img_w=256, img_h=256):
        # initialize image aspect ratio
        self.img_width = img_w
        self.img_height = img_h

        # define GAN

        # load model file, the blackbox detector is trained upon calling the init function.
        # In any case, it is trained externally
        self.blackbox = blackbox
        self.blackbox_detector = MalwareDetectionModels(self.img_width, self.img_height, self.blackbox)

        # create the generative model
        self.generator = self.build_generative_model()
        generator_optimizer = optimizers.Adam(lr=0.001)
        # self.generator.compile(optimizer=generator_optimizer, loss=losses.binary_crossentropy, metrics=['acc'])
        # perchè non si compila il generator?
        # https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch
        # / "The define_generator() function below defines the generator model, but intentionally does not compile it
        # as it is not trained directly, then returns the model."

        # create the discriminator
        self.substitute_detector = self.build_substitute_detector_model()
        discriminator_optimizer = optimizers.Adam(lr=0.001)
        # forse va cambiata la loss
        self.substitute_detector.compile(loss=losses.binary_crossentropy, optimizer=discriminator_optimizer,
                                         metrics=['accuracy'])

        # create the combined model (stack)
        # non ho capito ahhahah
        # The generator takes malware and noise as input and generates adversarial malware examples
        '''example = Input(shape=(self.apifeature_dims,))
        noise = Input(shape=(self.z_dims,))
        input = [example, noise]
        malware_examples = self.generator(input)'''

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
    @staticmethod
    def build_generative_model():
        # generator = models.Sequential() # Sequential model is not appropriate when: Your model has multiple inputs
        # or multiple outputs ref: https://keras.io/guides/sequential_model/
        mal_sample_input = Input(shape=(None,))
        noise_input = Input(shape=(None,))
        final_input = Concatenate(axis=1)([mal_sample_input, noise_input])
        current_output = layers.Dense(32 * 32 * 16, use_bias=False, input_shape=(100,))(final_input)
        current_output = layers.BatchNormalization()(current_output)
        current_output = layers.LeakyReLU()(current_output)

        current_output = layers.Reshape((32, 32, 16))(current_output)
        # assert current_output.output_shape == (None, 32, 32, 16)  # Note: None is the batch size

        current_output = layers.Conv2DTranspose(8, (5, 5), strides=(2, 2), padding='same', use_bias=False)(
            current_output)
        # assert current_output.output_shape == (None, 64, 64, 8)(current_output)
        current_output = layers.BatchNormalization()(current_output)
        current_output = layers.LeakyReLU()(current_output)

        current_output = layers.Conv2DTranspose(4, (5, 5), strides=(2, 2), padding='same', use_bias=False)(
            current_output)
        # assert current_output.output_shape == (None, 128, 128, 4)
        current_output = layers.BatchNormalization()(current_output)
        current_output = layers.LeakyReLU()(current_output)

        current_output = layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False,
                                                activation='tanh')(current_output)
        # assert current_output.output_shape == (None, 256, 256, 1)
        generator = Model(inputs=[noise_input, mal_sample_input], outputs=[current_output])
        print(generator.summary())

        return generator

    def build_substitute_detector_model(self):

        # it can be a convolutional network which works as a classifier
        substitute_detector = tf.keras.Sequential()
        substitute_detector.add(layers.Conv2D(64, (5, 5), activation='relu', strides=(2, 2), padding='same',
                                              input_shape=[self.img_width, self.img_height, 1]))

        substitute_detector.add(layers.MaxPooling2D(2, 2))

        substitute_detector.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        substitute_detector.add(layers.Conv2D(128, (5, 5), strides=(5, 5), padding='same'))
        substitute_detector.add(layers.MaxPooling2D(2, 2))

        substitute_detector.add(layers.Flatten())
        substitute_detector.add(layers.Dense(64, activation='relu'))
        substitute_detector.add(layers.Dropout(0.4))
        substitute_detector.add(layers.Dense(32, activation='relu'))
        substitute_detector.add(layers.Dropout(0.3))
        substitute_detector.add(layers.Dense(1, activation='sigmoid'))
        return substitute_detector

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def substitute_detector_loss(self, sd_on_benign, sd_on_adv, bb_on_benign, bb_on_adv):
        # the loss for the discriminator is computed as the sum of the loss related to real samples not being
        # classified correctly and the loss related to generated samples not being classified correctly
        benign_loss = self.cross_entropy(bb_on_benign, sd_on_benign)
        adv_loss = self.cross_entropy(bb_on_adv, sd_on_adv)
        total_loss = benign_loss + adv_loss
        return total_loss

    def generator_loss(self, sd_on_adv):
        return self.cross_entropy(tf.ones_like(sd_on_adv), sd_on_adv)

    EPOCHS = 50
    noise_dim = 100
    num_examples_to_generate = 16

    @tf.function
    def train_step(self, benign_images, malware_images):
        # in redefining how the function .fit() behaves for a model, the train_step(self,data) function needs to be
        # overridden ref https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
        noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])

        ''' "TensorFlow provides the tf.GradientTape API for automatic differentiation; 
                that is, computing the gradient of a computation with respect to some inputs, 
                usually tf.Variables. TensorFlow "records" relevant operations executed inside 
                the context of a tf.GradientTape onto a "tape". TensorFlow then uses that tape 
                to compute the gradients of a "recorded" computation using reverse mode differentiation."'''
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # with each step the generator produces a set of images
            adv_malware_samples = self.generator(malware_images, noise, training=True)

            # the discriminator makes predictions on real images first, then on adversarial samples. Each prediction
            # is stored into a variable and used to evaluate the discriminator loss
            
            bb_on_benign = self.blackbox(benign_images)
            bb_on_adv = self.blackbox(adv_malware_samples)

            sd_on_benign = self.substitute_detector(benign_images, training=True)
            sd_on_adv = self.substitute_detector(adv_malware_samples, training=True)

            gen_loss = self.generator_loss(sd_on_adv)
            disc_loss = self.substitute_detector_loss(sd_on_real, sd_on_adv, bb_on_real, bb_on_adv)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    def train_GAN(self, epochs):

        gan_training_set, gan_validation_set = UploadData.upload_training_validation_set('path/to/training_set_for_GAN')
        gan_test_set = UploadData.upload_test_set('path/to/test_set_for_GAN')
        for epoch in range(epochs):
            for image_batch in gan_training_set:
                self.train_step(image_batch)
