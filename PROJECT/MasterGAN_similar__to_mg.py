import tensorflow as tf
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Concatenate
from tensorflow.keras import layers
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, Sequential
from IPython import display
import tensorflow.keras.backend as K
import ImageProcessing
import math
from os.path import join as pjoin
#tf.config.run_functions_eagerly(True)

class MasterGAN:

    
    def __init__(self,loaded, blackbox, in_h, img_w=256, img_h=256):
        # initialize image aspect ratio
        self.img_width = img_w
        self.img_height = img_h
        self.EPOCHS = 20
        self.noise_dim = 100
        self.loaded = loaded
        self.num_examples_to_generate = 16
        self.in_height = in_h

        # CREATE AND COMPILE GENERATOR
        self.generator = self.build_generative_model()
        self.generator_optimizer = tf.keras.optimizers.Adam(lr=5e-5)
        #self.generator.compile(optimizer=self.generator_optimizer, loss=self.generator_loss, metrics=['acc'])
        
        # CREATE AND COMPILE DISCRIMINATOR
        self.substitute_detector = self.build_substitute_detector_model()
        self.substitute_detector_optimizer = tf.keras.optimizers.Adam(lr=1e-5)
        self.substitute_detector.compile(optimizer= self.substitute_detector_optimizer, loss= 'binary_crossentropy', metrics=['accuracy'],run_eagerly = True)
        
        # The generator takes malware and noise as input and generates adversarial malware examples
        example = Input(shape=(256,self.in_height,1))
        noise_ = Input(shape=(100))
        norm_input_ = Input(shape=(256,256,1))
        input = [example, noise_,norm_input_]
        malware_examples = self.generator(input)
        self.set_trainable(self.substitute_detector,False)
        validity = self.substitute_detector(malware_examples)

        

        #COMBINED MODEL, IT'S THE STACK OF THE DISCRIMINATOR AND THE GENERATOR. IN THIS MODEL THE DISCRIMINATOR SHALL NOT VE TRAINED
        self.combined = Model(input, validity)
        #self.combined = Sequential([self.generator, self.substitute_detector])
        self.combined.compile(optimizer=self.generator_optimizer , loss= 'binary_crossentropy', metrics=['accuracy'])
    


        # You will reuse this seed overtime (so it's easier)
        # to visualize progress in the animated GIF)
        self.seed = tf.random.normal([self.num_examples_to_generate, self.noise_dim])
        #####change to true
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        ''' add this option???
        self.same_train_data = same_train_data   # MalGAN and the black-boxdetector are trained on same or different training sets
        '''
        self.gen_loss_array,self.sub_det_loss_array = [],[]
        self.gen_acc_array,self.sub_det_acc_array = [],[]

        # load model file, the blackbox detector is trained upon calling the init function.
        # In any case, it is trained externally
        self.blackbox = blackbox
        #self.blackbox_detector = bb_detector
        #self.blackbox_detector = MalwareDetectionModels(self.img_width, self.img_height, self.blackbox)
        #self.bb_detector = MalwareDetectionModels(img_width, img_height, blackbox)

    
    def set_trainable(self, model, trainable):
        model.trainable = trainable
        for layer in model.layers: layer.trainable = trainable

    def build_substitute_detector_model(self):

      # it can be a convolutional network which works as a classifier
      substitute_detector = tf.keras.Sequential()
      substitute_detector.add(tf.keras.layers.experimental.preprocessing.Resizing(256, 256, interpolation="bilinear"))

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


   
    def build_generative_model(self):
    
        #ref https://www.tensorflow.org/tutorials/generative/pix2pix
        # generator = models.Sequential() # Sequential model is not appropriate when: Your model has multiple inputs
        # or multiple outputs ref: https://keras.io/guides/sequential_model/

        noise_input = Input(shape=(100)) #if we want the noise with size 256*256,we can use shape=(noise_dim,self.noise_dim)
        #mal_sample_input = Input(shape=(self.img_width,self.img_height))
        #final_input = Concatenate(axis = 1)([mal_sample_input, noise_input])
        noise_output = layers.Dense(16*16*16, use_bias=False)(noise_input)
        noise_output = layers.BatchNormalization()(noise_output)
        noise_output = layers.LeakyReLU()(noise_output)
        noise_output = layers.Reshape((16, 16, 16))(noise_output)
        # assert current_output.output_shape == (None, 32, 32, 16)  # Note: None is the batch size
        noise_output = layers.Conv2DTranspose(8, (5, 5), strides=(4, 4), padding='same', use_bias=False)(noise_output)
        # assert current_output.output_shape == (None, 64, 64, 8)(current_output)
        noise_output = layers.BatchNormalization()(noise_output)
        noise_output = layers.LeakyReLU()(noise_output)
        noise_output = layers.Conv2DTranspose(4, (5, 5), strides=(2, 2), padding='same', use_bias=False)(
        noise_output)
        # assert current_output.output_shape == (None, 128, 128, 4)
        noise_output = layers.BatchNormalization()(noise_output)
        noise_output = layers.LeakyReLU()(noise_output)
        noise_output = layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False,activation='sigmoid')(noise_output)
        # assert current_output.output_shape == (None, 256, 256, 1)
        norm_input = Input(shape = (256,256,1))
        #noise_output = layers.Flatten()(noise_output)
        noise_output = layers.Multiply(dtype=tf.int32)([noise_output, norm_input])

        malware_input = Input(shape = (256,self.in_height,1))
        
        final_output = Concatenate(axis=2)([malware_input, noise_output])

        generator = Model(inputs=[malware_input,noise_input,norm_input], outputs=[final_output])
        #print(generator.summary())
        tf.keras.utils.plot_model(
            generator,to_file='/home/qian/Masterproject/PLEASE', show_shapes=True, #show_dtype=True,
            show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96)

        
        return generator
        

    '''def checkpoint(self):
      checkpoint_dir = '/home/qian/Masterproject/PROJECT/saved_model/ckpt_GAN'
      checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
      checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                      substitute_detector_optimizer=self.substitute_detector_optimizer,
                                      generator=self.generator,
                                      discriminator=self.substitute_detector)
      return checkpoint,checkpoint_prefix'''


    def train(self,ben_ds, malware_sample,test_mal_path_exes, bb_on_real_):
      #ckpt,ckpt_prefix = self.checkpoint()
      norm_input = tf.ones(shape=[256,256,1])*255
      tmp=[]
      for i in range(0,64):
        tmp.append(norm_input)
      norm_input = tf.stack(tmp)

      for epoch in range(self.EPOCHS):
        print('\n Training epoch: '+ str(epoch))
        start = time.time()
        # • for each batch of malware paths and benign images we call the training step function
        for ben_batch, bb_on_real in zip(ben_ds,bb_on_real_):
            
          noise_generator = tf.random.Generator.from_non_deterministic_state()
          new_noise = noise_generator.normal(shape=[64,100])
            
          adv_noise_samples = self.generator.predict([malware_sample, new_noise, norm_input])
          
          bb_on_adv = self.blackbox.make_prediction(adv_noise_samples) # for each add flatten
          

          # Training
          gen_loss = 0
          gen_acc = 0
          dis_loss = 0
          dis_acc = 0 

          # Discriminator training
          self.set_trainable(self.substitute_detector, True)
          sub_loss_on_adv, sub_acc_on_adv = self.substitute_detector.train_on_batch(adv_noise_samples, np.around(bb_on_adv).astype(int))
          sub_loss_on_ben, sub_acc_on_ben = self.substitute_detector.train_on_batch(ben_batch,np.around(bb_on_real).astype(int))
          self.set_trainable(self.substitute_detector, False)
          sub_loss = (sub_loss_on_adv+sub_loss_on_ben)/2
          sub_acc = (sub_acc_on_adv+sub_acc_on_ben)/2
          

          # Generator training
          new_noise_1 = noise_generator.normal(shape=[64,100])
          gen_loss_now, gen_acc_now = self.combined.train_on_batch([malware_sample, new_noise_1,norm_input], np.zeros((64,1)))
          
          print('gen_loss = {0:.3f}, dis_loss = {1:.3f}, gen_acc = {2:.3f}, dis_acc = {3:.3f}'
                  .format(gen_loss_now, sub_loss, gen_acc_now, sub_acc))
                  
          self.gen_loss_array.append([gen_loss_now, epoch])
          self.sub_det_loss_array.append([sub_loss, epoch])
          self.gen_acc_array.append([gen_acc_now, epoch])
          self.sub_det_acc_array.append([sub_acc, epoch])
      
        np.save('/home/qian/Masterproject/Results/combined_loss.npy',np.array(self.gen_loss_array))
        np.save('/home/qian/Masterproject/Results/sub_det_loss.npy',np.array(self.sub_det_loss_array))
        np.save('/home/qian/Masterproject/Results/combined_acc.npy',np.array(self.gen_acc_array))
        np.save('/home/qian/Masterproject/Results/sub_det_acc.npy',np.array(self.sub_det_acc_array))

    def loss_plot(self):
    #plot the loss
      plt.figure()
      plt.plot(range(len(self.gen_loss_array) ), self.gen_loss_array, c='r', label='generator', linewidth=2)
      plt.plot(range(len(self.sub_det_loss_array)), self.sub_det_loss_array, c='g', linestyle='--', label='sub_discriminator', linewidth=2)
      plt.xlabel("Epoch")
      plt.ylabel("loss")
      plt.legend()
      plt.show()  

    def exe_to_img(self,path_to_exes):
        width = 256
        img_to_subdt='/home/qian/Masterproject/dataset/sub_temp_img/mid_img'
        self.clear_folder(img_to_subdt)
        i =0
        for path in path_to_exes:
            img_bin_array = ImageProcessing.readBytes(path)
            height = math.ceil(len(img_bin_array)/width)
            ImageProcessing.saveImg(pjoin(img_to_subdt,str(i))+'.png', img_bin_array, (width,height),'L')
            i = i+2
        
        mid_img_set = self.upload_pred_set('/home/qian/Masterproject/dataset/sub_temp_img')
        for only_batch in  mid_img_set:
            s = only_batch.shape[0]
            return only_batch  
          

    def upload_pred_set(self, DIRECTORY):
      COLOR_MODE = 'grayscale'        
      IMAGE_HEIGHT = 256
      IMAGE_WIDTH = 256
      BATCH_SIZE = 32
      SEED = 1337
      pred_set = keras.preprocessing.image_dataset_from_directory(
            DIRECTORY,
            label_mode = None,
            color_mode=COLOR_MODE,
            seed=SEED,
            interpolation="area",
            batch_size=BATCH_SIZE,
            image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
            shuffle=False
      )
      return pred_set

    def clear_folder(self,mid_dir):
      for f in os.listdir(mid_dir):
        os.remove(os.path.join(mid_dir, f))