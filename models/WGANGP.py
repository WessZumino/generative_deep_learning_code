
from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout, ZeroPadding2D, UpSampling2D
from keras.layers.merge import _Merge

from keras.models import Model, Sequential
from keras import backend as K
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint 
from keras.utils import plot_model
from keras.initializers import RandomNormal

from functools import partial

import numpy as np
import json
import os
import pickle
import matplotlib.pyplot as plt


class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((32, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class WGANGP():
    def __init__(self
        , input_dim
        , critic_conv_filters
        , critic_conv_kernel_size
        , critic_conv_strides
        , critic_conv_padding
        , critic_batch_norm_momentum
        , critic_activation
        , critic_dropout_rate
        , critic_learning_rate
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
        , optimiser
        , z_dim
        ):

        self.name = 'gan'

        self.input_dim = input_dim
        self.critic_conv_filters = critic_conv_filters
        self.critic_conv_kernel_size = critic_conv_kernel_size
        self.critic_conv_strides = critic_conv_strides
        self.critic_conv_padding = critic_conv_padding
        self.critic_batch_norm_momentum = critic_batch_norm_momentum
        self.critic_activation = critic_activation
        self.critic_dropout_rate = critic_dropout_rate
        self.critic_learning_rate = critic_learning_rate

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
        
        self.optimiser = optimiser

        self.z_dim = z_dim

        self.n_layers_critic = len(critic_conv_filters)
        self.n_layers_generator = len(generator_conv_t_filters)

        self.weight_init = RandomNormal(mean=0., stddev=0.02)

    
        self.critic = self._build_critic()
        self.generator = self._build_generator()

        self.model = self._build_adversarial()

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)

    def wasserstein(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def accuracy(self, y_true, y_pred):
        return K.mean(K.clip(K.sign(-y_true) * K.sign(y_pred),0,1))

    def _build_critic(self):

        ### THE critic
        critic_input = Input(shape=self.input_dim, name='critic_input')

        x = critic_input

        for i in range(self.n_layers_critic):

            x = Conv2D(
                filters = self.critic_conv_filters[i]
                , kernel_size = self.critic_conv_kernel_size[i]
                , strides = self.critic_conv_strides[i]
                , padding = self.critic_conv_padding
                , name = 'critic_conv_' + str(i)
                , kernel_initializer = self.weight_init
                )(x)

            if self.critic_batch_norm_momentum and i > 0:
                x = BatchNormalization(momentum = self.critic_batch_norm_momentum)(x)

            if self.critic_activation == 'leaky_relu':
                x = LeakyReLU(alpha = 0.2)(x)
            else:
                x = Activation(self.critic_activation)(x)

            if self.critic_dropout_rate:
                x = Dropout(rate = self.critic_dropout_rate)(x)

        x = Flatten()(x)
        
        critic_output = Dense(1, activation=None
        , kernel_initializer = self.weight_init
        )(x)

        critic = Model(critic_input, critic_output)

        return critic

    def _build_generator(self):

        ### THE generator

        generator_input = Input(shape=(self.z_dim,), name='generator_input')

        x = Dense(np.prod(self.generator_initial_dense_layer_size)
            , kernel_initializer = self.weight_init
            )(generator_input)
        if self.generator_batch_norm_momentum:
            x = BatchNormalization(momentum = self.generator_batch_norm_momentum)(x)

        if self.generator_activation == 'leaky_relu':
            x = LeakyReLU(alpha=0.2)(x)
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
                    , kernel_initializer = self.weight_init
                    )(x)

                if self.generator_batch_norm_momentum:
                    x = BatchNormalization(momentum = self.generator_batch_norm_momentum)(x)

                if self.generator_activation == 'leaky_relu':
                    x = LeakyReLU(alpha=0.2)(x)
                else:
                    x = Activation(self.generator_activation)(x)
                    
                
            else:
                x = Conv2DTranspose(
                    filters = self.generator_conv_t_filters[i]
                    , kernel_size = self.generator_conv_t_kernel_size[i]
                    , strides = self.generator_conv_t_strides[i]
                    , padding = self.generator_conv_t_padding
                    , name = 'generator_conv_t_' + str(i)
                    , kernel_initializer = self.weight_init
                    )(x)
                    
                x = Activation('tanh')(x)


        generator_output = x

        generator = Model(generator_input, generator_output)

        return generator


    def _build_adversarial(self):
        
        ### COMPILE critic

        if self.optimiser == 'adam':
            opti = Adam(lr=self.critic_learning_rate)
        elif self.optimiser == 'rmsprop':
            opti = RMSprop(lr=self.critic_learning_rate)
        else:
            opti = Adam(lr=self.critic_learning_rate)
        
        #-------------------------------
        # Construct Computational Graph
        #       for the Critic
        #-------------------------------

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        real_img = Input(shape=self.input_dim)

        # Noise input
        z_disc = Input(shape=(self.z_dim,))
        # Generate image based of noise (fake sample)
        fake_img = self.generator(z_disc)

        # critic determines validity of the real and fake images
        fake = self.critic(fake_img)
        valid = self.critic(real_img)

        # Construct weighted average between real and fake images
        # print(real_img.shape)
        # print(fake_img.shape)
        interpolated_img = RandomWeightedAverage()([real_img, fake_img])
        # Determine validity of weighted sample
        validity_interpolated = self.critic(interpolated_img)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                          averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.critic_model = Model(inputs=[real_img, z_disc],
                            outputs=[valid, fake, validity_interpolated])

        self.critic_model.compile(loss=[self.wasserstein,
                                              self.wasserstein,
                                              partial_gp_loss],
                                        optimizer=opti,
                                        loss_weights=[1, 1, 10])
        
        ### COMPILE THE FULL GAN

        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        model_input = Input(shape=(self.z_dim,))
        # Generate images based of noise
        img = self.generator(model_input)
        # Discriminator determines validity
        model_output = self.critic(img)
        # Defines generator model
        model = Model(model_input, model_output)

        if self.optimiser == 'adam':
            opti = Adam(lr = self.generator_learning_rate) 
        elif self.optimiser == 'rmsprop':
            opti = RMSprop(lr=self.generator_learning_rate)
        else:
            opti = Adam(lr = self.generator_learning_rate) 
        
        
        model.compile(optimizer=opti, loss=self.wasserstein, metrics=[self.accuracy])

        return model


    
    def train_critic(self, x_train, batch_size, using_generator):

        valid = -np.ones((batch_size,1))
        fake = np.ones((batch_size,1))
        dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty

        if using_generator:
            true_imgs = next(x_train)[0]
        else:
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            true_imgs = x_train[idx]
    
        noise = np.random.normal(0, 1, (batch_size, self.z_dim))

        d_loss = self.critic_model.train_on_batch([true_imgs, noise],
                                                                [valid, fake, dummy])
        return d_loss

    def train_generator(self, batch_size):
        valid = -np.ones((batch_size,1))
        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        return self.model.train_on_batch(noise, valid)


    def train(self, x_train, batch_size, epochs, run_folder, print_every_n_batches = 10
    , initial_epoch = 0
    , n_critic = 5
    , using_generator = False):

        d_losses = []
        g_losses = []

        d_accs = []
        g_accs = []

        for epoch in range(initial_epoch, initial_epoch + epochs):
            
            for _ in range(n_critic):

                d_loss = self.train_critic(x_train, batch_size, using_generator)
                d_acc = 0

            g_loss, g_acc = self.train_generator(batch_size)

            # Plot the progress
            print ("%d (%d, %d) [D loss: (%.1f)] [D acc: (%.3f)] [G loss: %.1f] [G acc: %.3f]" % (epoch, n_critic, 1, d_loss[0], d_acc, g_loss, g_acc))

            d_losses.append(d_loss)
            g_losses.append(g_loss)
            d_accs.append(d_acc)
            g_accs.append(g_acc)

            # If at save interval => save generated image samples
            if epoch % print_every_n_batches == 0:
                self.sample_images(initial_epoch + epoch, run_folder)

        # return d_losses_real, d_losses_fake, g_losses, d_accs_real, d_accs_fake, g_accs
        return d_losses, g_losses, d_accs, g_accs

    def sample_images(self, epoch, run_folder):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.z_dim))
        gen_imgs = self.generator.predict(noise)

        #Rescale images 0 - 1

        gen_imgs = 0.5 * (gen_imgs + 1)
        gen_imgs = np.clip(gen_imgs, 0, 1)

        fig, axs = plt.subplots(r, c)
        cnt = 0

        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(np.squeeze(gen_imgs[cnt, :,:,:]), cmap = 'gray_r')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(os.path.join(run_folder, "images/sample_%d.png" % epoch))
        plt.close()




    
    def plot_model(self, run_folder):
        plot_model(self.model, to_file=os.path.join(run_folder ,'viz/model.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.critic, to_file=os.path.join(run_folder ,'viz/critic.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.generator, to_file=os.path.join(run_folder ,'viz/generator.png'), show_shapes = True, show_layer_names = True)



            
    def save(self, folder):

            if not os.path.exists(folder):
                os.makedirs(folder)
                os.makedirs(os.path.join(folder, 'viz'))
                os.makedirs(os.path.join(folder, 'weights'))
                os.makedirs(os.path.join(folder, 'images'))

            with open(os.path.join(folder, 'params.pkl'), 'wb') as f:
                pickle.dump([
                    self.input_dim
                    , self.critic_conv_filters
                    , self.critic_conv_kernel_size
                    , self.critic_conv_strides
                    , self.critic_conv_padding
                    , self.critic_batch_norm_momentum
                    , self.critic_activation
                    , self.critic_dropout_rate
                    , self.critic_learning_rate
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