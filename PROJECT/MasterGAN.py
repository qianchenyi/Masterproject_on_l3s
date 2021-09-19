# from Models import MalwareDetectionModels
# from UploadData import UploadData
import tensorflow as tf
import numpy as np
import os
import tensorflow.keras as keras
from tensorflow.keras import layers, models, applications, optimizers, losses, metrics
from keras.layers.merge import Concatenate
from tensorflow.keras import layers
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from IPython import display

class MasterGAN:
    BUFFER_SIZE = 60000
    BATCH_SIZE = 256
    gen_loss_array,sub_det_loss_array = [],[]
    def __init__(self, blackbox, img_w=256, img_h=256):
        # initialize image aspect ratio
        self.img_width = img_w
        self.img_height = img_h
        self.EPOCHS = 50
        self.noise_dim = 256
        
        self.num_examples_to_generate = 16

        # You will reuse this seed overtime (so it's easier)
        # to visualize progress in the animated GIF)
        self.seed = tf.random.normal([self.num_examples_to_generate, self.noise_dim])
        
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        ''' add this option???
        self.same_train_data = same_train_data   # MalGAN and the black-boxdetector are trained on same or different training sets
        '''

        

        # load model file, the blackbox detector is trained upon calling the init function.
        # In any case, it is trained externally
        self.blackbox = blackbox
        #self.blackbox_detector = bb_detector
        #self.blackbox_detector = MalwareDetectionModels(self.img_width, self.img_height, self.blackbox)
        #self.bb_detector = MalwareDetectionModels(img_width, img_height, blackbox)


        ####### define GAN #########
        self.substitute_detector_optimizer = tf.keras.optimizers.Adam(lr=1e-5, beta_1=0.1)
        self.generator_optimizer = tf.keras.optimizers.Adam(lr=2e-4, beta_1=0.5)
        self.generator = self.build_generative_model()
        self.substitute_detector = self.build_substitute_detector_model()

        '''###load the dataset####
        uploader = UploadData()
        self.train_mal_set, self.val_mal_set = uploader.upload_train_mal_set()
        self.train_ben_set, self.val_ben_set = uploader.upload_train_ben_set()
        self.test_mal_set = uploader.upload_test_mal_set()
        self.test_ben_set = uploader.upload_test_ben_set()'''


   
    def build_generative_model(self):
        #ref https://www.tensorflow.org/tutorials/generative/pix2pix
        # generator = models.Sequential() # Sequential model is not appropriate when: Your model has multiple inputs
        # or multiple outputs ref: https://keras.io/guides/sequential_model/
        noise_input = Input(shape=(256,self.noise_dim)) #if we want the noise with size 256*256,we can use shape=(noise_dim,self.noise_dim)
        mal_sample_input = Input(shape=(self.img_width,self.img_height))
        final_input = Concatenate(axis = 1)([mal_sample_input, noise_input])
        print(final_input)
        current_output = layers.Dense(32, use_bias=False)(final_input)
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
        #print(generator.summary())
        tf.keras.utils.plot_model(
            generator,to_file='/content/data/model.png', show_shapes=True, show_dtype=True,
            show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96,
            layer_range=None)
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

        substitute_detector.add(layers.Dense(2, activation='sigmoid'))
        return substitute_detector

    
    def substitute_detector_loss(self, sd_on_real, sd_on_adv, bb_on_real, bb_on_adv):
        # the loss for the discriminator is computed as the sum of the loss related to real samples not being
        # classified correctly and the loss related to generated samples not being classified correctly
        real_loss = self.cross_entropy(bb_on_real, sd_on_real)
        fake_loss = self.cross_entropy(bb_on_adv, sd_on_adv)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, sd_on_adv):
        return self.cross_entropy(tf.ones_like(sd_on_adv), sd_on_adv)

    def checkpoint(self):
      checkpoint_dir = '/content/sample_data/'
      checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
      checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                      substitute_detector_optimizer=self.substitute_detector_optimizer,
                                      generator=self.generator,
                                      discriminator=self.substitute_detector)
      return checkpoint,checkpoint_prefix
    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    @tf.function
    def train_step(self,batch_mal, batch_ben):
        # in redefining how the function .fit() behaves for a model, the train_step(self,data) function needs to be
        # overridden ref https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
        img_batch_mal, lbls_mal = batch_mal
        img_batch_ben, lbls_ben = batch_ben
        img_nums= len(lbls_mal)
        new_noise = tf.random.normal([img_nums,self.img_width, self.img_height])
   

        ''' "TensorFlow provides the tf.GradientTape API for automatic differentiation; 
                that is, computing the gradient of a computation with respect to some inputs, 
                usually tf.Variables. TensorFlow "records" relevant operations executed inside 
                the context of a tf.GradientTape onto a "tape". TensorFlow then uses that tape 
                to compute the gradients of a "recorded" computation using reverse mode differentiation."'''
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # with each step the generator produces a set of images
            
            adv_malware_samples = self.generator([new_noise,img_batch_mal], training=True)

            # the discriminator makes predictions on real images first, then on adversarial samples. Each prediction
            # is stored into a variable and used to evaluate the discriminator loss
            bb_on_real = self.blackbox.model(img_batch_ben)
            sd_on_real = self.substitute_detector(img_batch_ben, training=True)
            bb_on_adv = self.blackbox.model(adv_malware_samples)
            sd_on_adv = self.substitute_detector(adv_malware_samples, training=True)

            gen_loss = self.generator_loss(sd_on_adv)
            disc_loss = self.substitute_detector_loss(sd_on_real, sd_on_adv, bb_on_real, bb_on_adv)
            gen_loss_array.append(gen_loss)
            sub_det_loss_array.append(disc_loss)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_sub_detector = disc_tape.gradient(disc_loss, self.substitute_detector.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.substitute_detector_optimizer.apply_gradients(
            zip(gradients_of_sub_detector, self.substitute_detector.trainable_variables))
    
    def train(self,ben_images, mal_images,test_mal_set):
      ckpt,ckpt_prefix = self.checkpoint()

      for epoch in range(self.EPOCHS):
        start = time.time()
        for batch_mal,batch_ben in zip(mal_images,ben_images):

          self.train_step(batch_mal,batch_ben)
        
        # Produce images for the GIF as you go
        display.clear_output(wait=True)
        self.generate_and_save_images(self.generator,epoch + 1,test_mal_set,self.seed)
                                

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
          ckpt.save(file_prefix = ckpt_prefix)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

      # Generate after the final epoch
      display.clear_output(wait=True)
      self.generate_and_save_images(self.generator,self.EPOCHS,test_mal_set,self.seed)

                      
                              

    def generate_and_save_images(self, generator, epoch, test_mal_set,noise):
      # Notice `training` is set to False.
      # This is so all layers run in inference mode (batchnorm).
      
      predictions = generator([test_mal_set,noise], training=False)

      fig = plt.figure(figsize=(4, 4))

      for i in range(predictions.shape[0]):
          plt.subplot(4, 4, i+1)
          plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
          plt.axis('off')

      plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
      plt.show()
    
    '''def __main__():
      ###load the dataset####
      uploader = UploadData()
      train_mal_set, self.val_mal_set = uploader.upload_train_mal_set()
      train_ben_set, self.val_ben_set = uploader.upload_train_ben_set()
      test_mal_set = uploader.upload_test_mal_set()
      test_ben_set = uploader.upload_test_ben_set()
      train(train_)'''
    def loss_plot():
    #plot the loss
      plt.figure()
      plt.plot(range(len(gen_loss_array) ), gen_loss_array, c='r', label='generator', linewidth=2)
      plt.plot(range(len(sub_det_loss_array)), sub_det_loss_array, c='g', linestyle='--', label='sub_discriminator', linewidth=2)
      plt.xlabel("Epoch")
      plt.ylabel("loss")
      plt.legend()
      plt.show()  