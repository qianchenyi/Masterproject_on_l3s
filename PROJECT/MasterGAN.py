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
        self.EPOCHS = 200
        self.noise_dim = 65536
        self.loaded = loaded
        self.num_examples_to_generate = 16
        self.in_height = in_h
        self.noise_generator = tf.random.Generator.from_seed(2)

        
        # CREATE AND COMPILE DISCRIMINATOR
        self.substitute_detector = self.build_substitute_detector_model()
        self.substitute_detector_optimizer = tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.1)
        self.substitute_detector.compile(optimizer= self.substitute_detector_optimizer, loss= 'binary_crossentropy', metrics=['accuracy'],run_eagerly = True)
        
        
        # CREATE GENERATOR
        self.generator = self.build_generative_model()
        self.generator_optimizer = tf.keras.optimizers.Adam(lr=5e-3, beta_1=0.5)

        
        # The generator takes malware and noise as input and generates adversarial malware examples
        example = Input(shape=(self.in_height,256, 1))
        noise_ = Input(shape=(65536))
        input = [example, noise_]
        malware_examples = self.generator(input)
        self.set_trainable(self.generator,True)
        self.set_trainable(self.substitute_detector,False)
        validity = self.substitute_detector(malware_examples)

        
        #CREATE AND COMPILE THE COMBINED MODEL
        self.combined = Model(input, validity)
        self.combined.compile(optimizer=self.generator_optimizer , loss= 'binary_crossentropy', metrics=['accuracy'])
    

        self.seed = tf.random.normal([self.num_examples_to_generate, self.noise_dim])
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        
        # metrics array for each epoch 
        self.gen_loss_array,self.sub_det_loss_array = [],[]
        self.gen_acc_array,self.sub_det_acc_array = [],[]
        # metrics array for each batch 
        self.gen_loss_batch_array, self.sub_det_loss_batch_array = [],[]
        self.gen_acc_batch_array, self.sub_det_acc_batch_array = [],[]
        self.bb_on_adv_array = []
 

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
        substitute_detector.add(layers.BatchNormalization())
        substitute_detector.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        substitute_detector.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        substitute_detector.add(layers.MaxPooling2D(2, 2))


        substitute_detector.add(layers.Flatten())
        substitute_detector.add(layers.Dense(64, activation='relu'))
        substitute_detector.add(layers.Dropout(0.2))
        substitute_detector.add(layers.Dense(32, activation='relu'))
        substitute_detector.add(layers.Dropout(0.2))
        substitute_detector.add(layers.Dense(16, activation='relu'))
        substitute_detector.add(layers.Dropout(0.2))

        substitute_detector.add(layers.Dense(1, activation='sigmoid'))
        

        return substitute_detector


   
    def build_generative_model(self):
    
        noise_input = Input(shape=(65536)) 
        noise_output = layers.Dense(16*16*16, use_bias=False)(noise_input)
        noise_output = layers.BatchNormalization()(noise_output)
        noise_output = layers.LeakyReLU()(noise_output)
        noise_output = layers.Reshape((16, 16, 16))(noise_output)
        noise_output = layers.Conv2DTranspose(8, (5, 5), strides=(1, 1), padding='same', use_bias=False)(noise_output)
        noise_output = layers.BatchNormalization()(noise_output)
        noise_output = layers.LeakyReLU()(noise_output)
        noise_output = layers.Conv2DTranspose(8, (5, 5), strides=(4, 4), padding='same', use_bias=False)(
        noise_output)
        noise_output = layers.BatchNormalization()(noise_output)
        noise_output = layers.LeakyReLU()(noise_output)
        noise_output = layers.Conv2DTranspose(8, (5, 5), strides=(1, 2), padding='same', use_bias=False)(
        noise_output)
        noise_output = layers.BatchNormalization()(noise_output)
        noise_output = layers.LeakyReLU()(noise_output)
        '''noise_output = layers.Conv2DTranspose(4, (3, 3), strides=(2, 2), padding='same', use_bias=False)(
        noise_output)
        noise_output = layers.BatchNormalization()(noise_output)
        noise_output = layers.LeakyReLU()(noise_output)'''
        noise_output = layers.Conv2DTranspose(32, (3, 3), strides=(2, 1), padding='same', use_bias=False)(
        noise_output)
        noise_output = layers.BatchNormalization()(noise_output)
        noise_output = layers.LeakyReLU()(noise_output)
        
        noise_output = layers.Conv2DTranspose(1, (5, 5), strides=(1, 2), padding='same', use_bias=False, activation='tanh')(noise_output)

        old_min = tf.reduce_min(noise_output)
        old_max = tf.reduce_max(noise_output)
        noise_output = ((noise_output - old_min)*(255-0))/(old_max-old_min)
        #noise_output = tf.keras.layers.experimental.preprocessing.Resizing( self.in_height, 256, interpolation="bilinear",)(noise_output)
        #noise_output = ((noise_output+1)*255)/2
        malware_input = Input(shape = (self.in_height,256,1))
        final_output = Concatenate(axis=1)([malware_input, noise_output])

        generator = Model(inputs=[malware_input,noise_input], outputs=[final_output])
        #print(generator.summary())
        '''tf.keras.utils.plot_model(
            generator,to_file='/home/qian/Masterproject/Results/noise128/gen.pdf', show_shapes=True, 
            show_layer_names=True)'''

        
        return generator
        

    '''def checkpoint(self):
        checkpoint_dir = '/home/qian/Masterproject/PROJECT/saved_model/ckpt_GAN'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                      substitute_detector_optimizer=self.substitute_detector_optimizer,
                                      combined=self.combined,
                                      discriminator=self.substitute_detector)
        return checkpoint_dir,checkpoint_prefix,checkpoint'''


    def train(self,ben_ds, malware_sample,test_mal_path_exes, bb_on_real_):
        # ckpt_dir,ckpt_prefix,ckpt = self.checkpoint()
        # ckpt.restore(tf.train.latest_checkpoint(ckpt_dir))
        for epoch in range(self.EPOCHS):
            print('\n Training epoch: '+ str(epoch+1)+' of '+str(self.EPOCHS))
            bb_on_adv_epoch = None

            # • for each batch of malware paths and benign images we call the training step function
            ## Training metrics ##
            gen_loss = 0
            gen_acc = 0
            sub_det_loss = 0
            sub_det_acc = 0 
            counter = 0


            for ben_batch, bb_on_real in zip(ben_ds,bb_on_real_):      
                counter = counter + 1
                new_noise = self.noise_generator.normal(shape=[64,65536])
                adv_noise_samples = self.generator.predict([malware_sample, new_noise])
                bb_on_adv = self.blackbox.make_prediction(adv_noise_samples,epoch) # for each add flatten
                bb_on_adv_epoch = tf.identity(bb_on_adv, name=None)



            # Discriminator training
                self.set_trainable(self.substitute_detector, True)
                sub_loss_now_adv, sub_acc_now_adv = self.substitute_detector.train_on_batch(adv_noise_samples, np.around(bb_on_adv).astype(int))
                sub_loss_now_ben, sub_acc_now_ben = self.substitute_detector.train_on_batch(ben_batch,np.around(bb_on_real).astype(int))
                self.set_trainable(self.substitute_detector, False)
                sub_loss_now = (sub_loss_now_adv+sub_loss_now_ben)/2
                sub_acc_now = (sub_acc_now_adv+sub_acc_now_ben)/2
                #print(self.substitute_detector.summary())
                #tf.keras.utils.plot_model(
                #    self.substitute_detector,to_file='/home/qian/Masterproject/Results/noise128/dis.pdf', show_shapes=True, 
                #    show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96)

            # Generator training
                new_noise_1 = self.noise_generator.normal(shape=[64,65536])
                gen_loss_now, gen_acc_now = self.combined.train_on_batch([malware_sample, new_noise_1], np.zeros((64,1)))


                gen_loss = gen_loss + gen_loss_now
                gen_acc = gen_acc + gen_acc_now
                sub_det_loss = sub_det_loss + sub_loss_now
                sub_det_acc =  sub_det_acc + sub_acc_now

                # save log for current batch
                log = '\n gen_loss_now = {0:.3f}, dis_loss_now = {1:.3f}, gen_acc_now = {2:.3f}, dis_acc_now = {3:.3f}'.format(gen_loss_now, sub_loss_now, gen_acc_now, sub_acc_now)
                with open("/home/qian/Masterproject/Results/noise128/log_by_batch.txt", "a") as myfile:
                    myfile.write(log)  

                # append metrics for the current batch
                self.gen_loss_batch_array.append(gen_loss_now)
                self.sub_det_loss_batch_array.append(sub_loss_now)
                self.gen_acc_batch_array.append(gen_acc_now)
                self.sub_det_acc_batch_array.append(sub_acc_now)
                np.save('/home/qian/Masterproject/Results/noise128/combined_loss_batch.npy',np.array(self.gen_loss_batch_array))
                np.save('/home/qian/Masterproject/Results/noise128/sub_det_loss_batch.npy',np.array(self.sub_det_loss_batch_array))
                np.save('/home/qian/Masterproject/Results/noise128/combined_acc_batch.npy',np.array(self.gen_acc_batch_array))
                np.save('/home/qian/Masterproject/Results/noise128/sub_det_acc_batch.npy',np.array(self.sub_det_acc_batch_array))
                np.save('/home/qian/Masterproject/Results/noise128/bb_on_adv.npy',np.array(self.bb_on_adv_array))
            


            # save the metrics
            self.gen_loss_array.append((gen_loss/counter))
            self.sub_det_loss_array.append((sub_det_loss/counter))
            self.gen_acc_array.append((gen_acc/counter))
            self.sub_det_acc_array.append((sub_det_acc/counter))
            self.bb_on_adv_array.append(bb_on_adv_epoch)
            
            log_by_epoch = '\n gen_loss = {0:.3f}, gen_acc = {1:.3f}, sub_det_loss = {2:.3f}, sub_det_acc = {3:.3f}'.format(gen_loss/counter, gen_acc/counter, sub_det_loss/counter, sub_det_acc/counter)
            with open("/home/qian/Masterproject/Results/noise128/log_by_epoch.txt", "a") as myfile:
                    myfile.write(log_by_epoch) 
            print(log_by_epoch)

            np.save('/home/qian/Masterproject/Results/noise128/combined_loss_epoch.npy',np.array(self.gen_loss_array))
            np.save('/home/qian/Masterproject/Results/noise128/sub_det_loss_epoch.npy',np.array(self.sub_det_loss_array))
            np.save('/home/qian/Masterproject/Results/noise128/combined_acc_epoch.npy',np.array(self.gen_acc_array))
            np.save('/home/qian/Masterproject/Results/noise128/sub_det_acc_epoch.npy',np.array(self.sub_det_acc_array))
        self.generator.save('/home/qian/Masterproject/PROJECT/saved_model/generator_model.h5')
        self.substitute_detector.save('/home/qian/Masterproject/PROJECT/saved_model/sub_detector.h5')
    

 