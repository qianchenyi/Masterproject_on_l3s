import tensorflow as tf
import numpy as np
import os
import time
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Concatenate
from tensorflow.keras import layers
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from os.path import join as pjoin
#tf.config.run_functions_eagerly(True)

class MasterGAN:

    
    def __init__(self,loaded, blackbox, in_h, img_w=256, img_h=256):
        self.img_width = img_w
        self.img_height = img_h
        self.EPOCHS = 450
        self.noise_dim = 100
        self.loaded = loaded
        self.num_examples_to_generate = 16
        self.in_height = in_h
        self.noise_generator = tf.random.Generator.from_non_deterministic_state()

        
        # CREATE AND COMPILE DISCRIMINATOR
        self.substitute_detector = self.build_substitute_detector_model()
        self.substitute_detector_optimizer = tf.keras.optimizers.Adam(lr=1e-6, beta_1=0.1)
        self.substitute_detector.compile(optimizer= self.substitute_detector_optimizer, loss= 'binary_crossentropy', metrics=['accuracy'],run_eagerly = True)
        # CREATE AND COMPILE GENERATOR
        self.generator = self.build_generative_model()
        self.generator_optimizer = tf.keras.optimizers.Adam(lr=3e-5, beta_1=0.5)
        #self.generator.compile(optimizer=self.generator_optimizer, loss=self.generator_loss, metrics=['acc'])
        
        # The generator takes malware and noise as input and generates adversarial malware examples
        example = Input(shape=(self.in_height,256, 1))
        noise_ = Input(shape=(100))
        input = [example, noise_]
        malware_examples = self.generator(input)
        self.set_trainable(self.generator,True)
        self.set_trainable(self.substitute_detector,False)
        validity = self.substitute_detector(malware_examples)

        

        self.combined = Model(input, validity)
        self.combined.compile(optimizer=self.generator_optimizer , loss= 'binary_crossentropy', metrics=['accuracy'])
    

        self.seed = tf.random.normal([self.num_examples_to_generate, self.noise_dim])
        #####change to true
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        self.gen_loss_array,self.sub_det_loss_array = [],[]
        self.gen_acc_array,self.sub_det_acc_array = [],[]

        self.blackbox = blackbox

    
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
    
        noise_input = Input(shape=(100)) 

        noise_output = layers.Dense(16*16*16, use_bias=False)(noise_input)
        noise_output = layers.Reshape((16, 16, 16))(noise_output)
        noise_output =tf.keras.layers.Conv2D(32, (3,3), strides=2, padding='same', use_bias=False)(noise_output)
        noise_output = layers.LeakyReLU()(noise_output)
        noise_output =tf.keras.layers.Conv2D(64, (5,5), strides=2, padding='same', use_bias=False)(noise_output)
        noise_output = layers.BatchNormalization()(noise_output)
        noise_output = layers.LeakyReLU()(noise_output)
        noise_output = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same', use_bias=False)(noise_output)
        noise_output = layers.BatchNormalization()(noise_output)
        noise_output = layers.LeakyReLU()(noise_output)
        #noise_output =tf.keras.layers.Conv2D(16, (4,4), strides=(2,2), padding='same', kernel_initializer=initializer, use_bias=False)(noise_output)
        
        noise_output = tf.keras.layers.Conv2DTranspose(1, 4,
                                                strides=2,
                                                padding='same',
                                                activation='tanh')(noise_output)

        old_min = tf.reduce_min(noise_output)
        old_max = tf.reduce_max(noise_output)
        noise_output = tf.keras.layers.experimental.preprocessing.Resizing(256, 256, interpolation="gaussian")(noise_output)
        noise_output = ((noise_output - old_min)*(255-0))/(old_max-old_min)

        #noise_output = tf.keras.layers.experimental.preprocessing.Resizing( self.in_height, 256, interpolation="bilinear",)(noise_output)
        #noise_output = ((noise_output+1)*255)/2
        malware_input = Input(shape = (self.in_height,256,1))
        final_output = Concatenate(axis=1)([malware_input, noise_output])

        generator = Model(inputs=[malware_input,noise_input], outputs=[final_output])
        #print(generator.summary())
        '''tf.keras.utils.plot_model(s
            generator,to_file='/home/fasci/master_project/dataset/gen.png', show_shapes=True, 
            show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96)'''

        
        return generator
            

        '''def checkpoint(self):
        checkpoint_dir = '/home/fasci/master_project/saved_model/ckp3t_GAN'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        
        checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                        substitute_detector_optimizer=self.substitute_detector_optimizer,
                                        generator=self.generator,
                                        discriminator=self.substitute_detector)
        return checkpoint,checkpoint_prefix'''


    def train(self,ben_ds, malware_sample,test_mal_path_exes, bb_on_real_):
      #ckpt,ckpt_prefix = self.checkpoint()
        for epoch in range(self.EPOCHS):
            print('\n Training epoch: '+ str(epoch+1)+' of '+str(self.EPOCHS))
            start = time.time()
            tr=0
            # Training
            gen_loss = 0
            gen_acc = 0
            dis_loss = 0
            dis_acc = 0 
            # • for each batch of malware paths and benign images we call the training step function
            counter = 0
            for ben_batch, bb_on_real in zip(ben_ds,bb_on_real_):
                
                counter = counter + 1
                new_noise = self.noise_generator.normal(shape=[64,100])
                adv_noise_samples = self.generator.predict([malware_sample, new_noise])
                
                bb_on_adv = self.blackbox.make_prediction(adv_noise_samples) # for each add flatten
                tr = tr + 1 
                #print(bb_on_adv)

                

                # Discriminator training
                self.set_trainable(self.substitute_detector, True)
                sub_loss_on_adv, sub_acc_on_adv = self.substitute_detector.train_on_batch(adv_noise_samples, np.around(bb_on_adv).astype(int))
                sub_loss_on_ben, sub_acc_on_ben = self.substitute_detector.train_on_batch(ben_batch,np.around(bb_on_real).astype(int))
                self.set_trainable(self.substitute_detector, False)
                sub_loss = (sub_loss_on_adv+sub_loss_on_ben)/2
                sub_acc = (sub_acc_on_adv+sub_acc_on_ben)/2
                

                # Generator training
                new_noise_1 = self.noise_generator.normal(shape=[64,100])
                gen_loss_now, gen_acc_now = self.combined.train_on_batch([malware_sample, new_noise_1], np.zeros((64,1)))
                
                
                gen_loss = gen_loss + gen_loss_now
                gen_acc = gen_acc + gen_acc_now
                dis_loss = dis_loss + sub_loss
                dis_acc =  dis_acc + sub_acc

                        
                self.gen_loss_array.append((gen_loss_now,epoch))
                self.sub_det_loss_array.append((sub_loss,epoch))
                self.gen_acc_array.append((gen_acc_now,epoch))
                self.sub_det_acc_array.append((sub_acc,epoch))
                print('\n gen_loss_batch = {0:.3f}, gen_acc_batch = {1:.3f}, dis_loss_batch = {2:.3f}, dis_acc_batch = {3:.3f}'.format(gen_loss_now, gen_acc_now, sub_loss, sub_acc))


                log = '\n gen_loss = {0:.3f}, gen_acc = {1:.3f}, dis_loss = {2:.3f}, dis_acc = {3:.3f}'.format(gen_loss/counter, gen_acc/counter, dis_loss/counter, dis_acc/counter)
                
        
                with open("/home/qian/Masterproject/Results/log.txt", "a") as myfile:
                    myfile.write(log) 
            
            
        np.save('/home/qian/Masterproject/Results/combined_loss.npy',np.array(self.gen_loss_array))
        np.save('/home/qian/Masterproject/Results/sub_det_loss.npy',np.array(self.sub_det_loss_array))
        np.save('/home/qian/Masterproject/Results/combined_acc.npy',np.array(self.gen_acc_array))
        np.save('/home/qian/Masterproject/Results/sub_det_acc.npy',np.array(self.sub_det_acc_array))


    

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