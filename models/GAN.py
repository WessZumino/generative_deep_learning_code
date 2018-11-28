
from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout, ZeroPadding2D, UpSampling2D
from keras.layers.merge import _Merge

from keras.models import Model, Sequential
from keras import backend as K
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint 
from keras.utils import plot_model
from keras.initializers import RandomNormal

from utils.callbacks import CustomCallback, step_decay_schedule

import numpy as np
import json
import os
import pickle
import matplotlib.pyplot as plt


class GAN():
    def __init__(self
        , input_dim
        , discriminator_conv_filters
        , discriminator_conv_kernel_size
        , discriminator_conv_strides
        , discriminator_conv_padding
        , discriminator_batch_norm_momentum
        , discriminator_activation
        , discriminator_dropout_rate
        , discriminator_learning_rate
        , generator_initial_dense_layer_size
        , generator_use_upsampling
        , generator_conv_t_filters
        , generator_conv_t_kernel_size
        , generator_conv_t_strides
        , generator_conv_t_padding
        , generator_batch_norm_momentum
        , generator_activation
        , generator_dropout_rate
        , generator_learning_rate
        , z_dim
        ):

        self.name = 'gan'

        self.input_dim = input_dim
        self.discriminator_conv_filters = discriminator_conv_filters
        self.discriminator_conv_kernel_size = discriminator_conv_kernel_size
        self.discriminator_conv_strides = discriminator_conv_strides
        self.discriminator_conv_padding = discriminator_conv_padding
        self.discriminator_batch_norm_momentum = discriminator_batch_norm_momentum
        self.discriminator_activation = discriminator_activation
        self.discriminator_dropout_rate = discriminator_dropout_rate
        self.discriminator_learning_rate = discriminator_learning_rate

        self.generator_initial_dense_layer_size = generator_initial_dense_layer_size
        self.generator_use_upsampling = generator_use_upsampling
        self.generator_conv_t_filters = generator_conv_t_filters
        self.generator_conv_t_kernel_size = generator_conv_t_kernel_size
        self.generator_conv_t_strides = generator_conv_t_strides
        self.generator_conv_t_padding = generator_conv_t_padding
        self.generator_batch_norm_momentum = generator_batch_norm_momentum
        self.generator_activation = generator_activation
        self.generator_dropout_rate = generator_dropout_rate
        self.generator_learning_rate = generator_learning_rate

        self.z_dim = z_dim

        self.n_layers_discriminator = len(discriminator_conv_filters)
        self.n_layers_generator = len(generator_conv_t_filters)

        self.weight_init = RandomNormal(mean=0., stddev=0.02)

        self.discriminator = self._build_discriminator()
        self.generator = self._build_generator()

        self.model = self._build_adversarial()

    

    def _build_discriminator(self):

        ### THE discriminator
        discriminator_input = Input(shape=self.input_dim, name='discriminator_input')

        x = discriminator_input

        for i in range(self.n_layers_discriminator):

            x = Conv2D(
                filters = self.discriminator_conv_filters[i]
                , kernel_size = self.discriminator_conv_kernel_size[i]
                , strides = self.discriminator_conv_strides[i]
                , padding = self.discriminator_conv_padding
                , name = 'discriminator_conv_' + str(i)
                )(x)

            if self.discriminator_batch_norm_momentum:
                x = BatchNormalization(momentum = self.discriminator_batch_norm_momentum)(x)

            if self.discriminator_activation == 'leaky_relu':
                x = LeakyReLU(x)
            else:
                x = Activation(self.discriminator_activation)(x)

            if self.discriminator_dropout_rate:
                x = Dropout(rate = self.discriminator_dropout_rate)(x)

        x = Flatten()(x)
        
        discriminator_output = Dense(1, activation='sigmoid')(x)

        discriminator = Model(discriminator_input, discriminator_output)

        return discriminator

    def _build_generator(self):

        ### THE generator

        generator_input = Input(shape=(self.z_dim,), name='generator_input')

        x = Dense(np.prod(self.generator_initial_dense_layer_size))(generator_input)
        if self.generator_batch_norm_momentum:
            x = BatchNormalization(momentum = self.generator_batch_norm_momentum)(x)

        if self.generator_activation == 'leaky_relu':
            x = LeakyReLU(x)
        else:
            x = Activation(self.generator_activation)(x)

        x = Reshape(self.generator_initial_dense_layer_size)(x)

        if self.generator_dropout_rate:
            x = Dropout(rate = self.generator_dropout_rate)(x)

        for i in range(self.n_layers_generator):

            if i < self.n_layers_generator - 1:
                if self.generator_use_upsampling[i]:
                    x = UpSampling2D()(x)

                x = Conv2DTranspose(
                    filters = self.generator_conv_t_filters[i]
                    , kernel_size = self.generator_conv_t_kernel_size[i]
                    , strides = self.generator_conv_t_strides[i]
                    , padding = self.generator_conv_t_padding
                    , name = 'generator_conv_t_' + str(i)
                    )(x)

                if self.generator_batch_norm_momentum:
                    x = BatchNormalization(momentum = self.generator_batch_norm_momentum)(x)

                if self.generator_activation == 'leaky_relu':
                    x = LeakyReLU(x)
                else:
                    x = Activation(self.generator_activation)(x)
                    
                
            else:
                x = Conv2DTranspose(
                    filters = self.generator_conv_t_filters[i]
                    , kernel_size = self.generator_conv_t_kernel_size[i]
                    , strides = self.generator_conv_t_strides[i]
                    , padding = self.generator_conv_t_padding
                    , name = 'generator_conv_t_' + str(i)
                    )(x)
                    
                x = Activation('tanh')(x)


        generator_output = x

        generator = Model(generator_input, generator_output)

        return generator


    def _build_adversarial(self):
        
        ### COMPILE DISCRIMINATOR
    
        self.discriminator.compile(optimizer=RMSprop(lr=self.discriminator_learning_rate), loss = 'binary_crossentropy',  metrics = ['accuracy'])
        
        ### COMPILE THE FULL GAN

        self.discriminator.trainable = False
        for l in self.discriminator.layers:
            l.trainable = False

        model_input = Input(shape=(self.z_dim,), name='model_input')
        model_output = self.discriminator(self.generator(model_input))

        model = Model(model_input, model_output)

        model.compile(optimizer=RMSprop(lr=self.generator_learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

        return model


    def save(self, folder):

        if not os.path.exists(folder):
            os.makedirs(folder)
            os.makedirs(os.path.join(folder, 'viz'))
            os.makedirs(os.path.join(folder, 'weights'))
            os.makedirs(os.path.join(folder, 'images'))

        with open(os.path.join(folder, 'params.pkl'), 'wb') as f:
            pickle.dump([
                self.input_dim
                , self.discriminator_conv_filters
                , self.discriminator_conv_kernel_size
                , self.discriminator_conv_strides
                , self.discriminator_conv_padding
                , self.discriminator_batch_norm_momentum
                , self.discriminator_activation
                , self.discriminator_dropout_rate
                , self.discriminator_learning_rate
                , self.generator_initial_dense_layer_size
                , self.generator_use_upsampling
                , self.generator_conv_t_filters
                , self.generator_conv_t_kernel_size
                , self.generator_conv_t_strides
                , self.generator_conv_t_padding
                , self.generator_batch_norm_momentum
                , self.generator_activation
                , self.generator_dropout_rate
                , self.generator_learning_rate
                , self.z_dim
                ], f)

        self.plot_model(folder)


    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    def train(self, x_train, batch_size, epochs, run_folder, print_every_n_batches = 50, initial_epoch = 0):
        d_losses = []
        g_losses = []
        d_acc = []
        g_acc = []

        for epoch in range(initial_epoch, initial_epoch + epochs):

            idx = np.random.randint(0, x_train.shape[0], batch_size)
            true_imgs = x_train[idx]
            
            noise = np.random.normal(0, 1, (batch_size, self.z_dim))
            gen_imgs = self.generator.predict(noise)

            x = np.concatenate((true_imgs,gen_imgs))
            y = np.ones([2*batch_size,1])
            y[batch_size:,:] = 0

            d_loss = self.discriminator.train_on_batch(x,y)

            noise = np.random.normal(0, 1, (batch_size, self.z_dim))
            y = np.ones([batch_size,1])
            g_loss = self.model.train_on_batch(noise, y)

            # Plot the progress
            print ("%d [D loss: %f] [D acc: %f] [G loss: %f] [G acc: %f]" % (epoch, d_loss[0], d_loss[1], g_loss[0], g_loss[1]))

            d_losses.append(d_loss[0])
            g_losses.append(g_loss[0])
            d_acc.append(d_loss[1])
            g_acc.append(g_loss[1])

            # If at save interval => save generated image samples
            if epoch % print_every_n_batches == 0:
                self.sample_images(initial_epoch + epoch, run_folder)

        return d_losses, g_losses, d_acc, g_acc

    def sample_images(self, epoch, run_folder):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.z_dim))
        gen_imgs = self.generator.predict(noise)

        #Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 1

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray_r')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(os.path.join(run_folder, "images/sample_%d.png" % epoch))
        plt.close()




    
    def plot_model(self, run_folder):
        plot_model(self.model, to_file=os.path.join(run_folder ,'viz/model.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.discriminator, to_file=os.path.join(run_folder ,'viz/discriminator.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.generator, to_file=os.path.join(run_folder ,'viz/generator.png'), show_shapes = True, show_layer_names = True)



        


        

        


