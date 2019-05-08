#
# Based on https://github.com/tatsy/keras-generative
#

import os
import sys
import math
import time
import numpy as np

import keras
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Lambda, Reshape, Concatenate
from keras.layers import Activation, LeakyReLU, ELU, Dropout
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, BatchNormalization
from keras.optimizers import Adam
from keras.models import load_model
from keras import backend as K

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


output = "outputs"
latent_dim = 3
epochs = 1000
batch_size = 64
intermediate_dim = 100
beta = 4

def sample_normal(args):
    z_avg, z_log_var = args
    batch_size = K.shape(z_avg)[0]
    z_dims = K.shape(z_avg)[1]
    eps = K.random_normal(shape=(batch_size, z_dims), mean=0.0, stddev=1.0)
    return z_avg + K.exp(z_log_var / 2.0) * eps

def zero_loss(y_true, y_pred):
    return K.zeros_like(y_true)

def set_trainable(model, train):
    """
    Enable or disable training for the model
    """
    model.trainable = train
    for l in model.layers:
        l.trainable = train


class SampleNormal(Layer):
    __name__ = 'sample_normal'

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(SampleNormal, self).__init__(**kwargs)

    def _sample_normal(self, z_avg, z_log_var):
        batch_size = K.shape(z_avg)[0]
        z_dims = K.shape(z_avg)[1]
        eps = K.random_normal(shape=K.shape(z_avg), mean=0.0, stddev=1.0)
        return z_avg + K.exp(z_log_var / 2.0) * eps

    def call(self, inputs):
        z_avg = inputs[0]
        z_log_var = inputs[1]
        return self._sample_normal(z_avg, z_log_var)

class MinibatchDiscrimination(Layer):
    __name__ = 'minibatch_discrimination'

    def __init__(self, kernels=50, dims=5, **kwargs):
        super(MinibatchDiscrimination, self).__init__(**kwargs)
        self.kernels = kernels
        self.dims = dims

    def build(self, input_shape):
        assert len(input_shape) == 2
        self.W = self.add_weight(name='kernel',
                                 shape=(input_shape[1], self.kernels * self.dims),
                                 initializer='uniform',
                                 trainable=True)

    def call(self, inputs):
        Ms = K.dot(inputs, self.W)
        Ms = K.reshape(Ms, (-1, self.kernels, self.dims))
        x_i = K.reshape(Ms, (-1, self.kernels, 1, self.dims))
        x_j = K.reshape(Ms, (-1, 1, self.kernels, self.dims))
        x_i = K.repeat_elements(x_i, self.kernels, 2)
        x_j = K.repeat_elements(x_j, self.kernels, 1)
        norm = K.sum(K.abs(x_i - x_j), axis=3)
        Os = K.sum(K.exp(-norm), axis=2)
        return Os

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.kernels)

def BasicConvLayer(
    filters,
    kernel_size=(5, 5),
    padding='same',
    strides=(1, 1),
    bnorm=True,
    dropout=0.0,
    activation='leaky_relu'):

    def fun(inputs):
        kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        bias_init = keras.initializers.Zeros()

        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   kernel_initializer=kernel_init,
                   bias_initializer=bias_init,
                   padding=padding)(inputs)

        if bnorm:
            x = BatchNormalization()(x)

        if activation == 'leaky_relu':
            x = LeakyReLU(0.1)(x)
        elif activation == 'elu':
            x = ELU()(x)
        else:
            x = Activation(activation)(x)

        if dropout > 0.0:
            x = Dropout(dropout)(x)

        return x

    return fun

def BasicDeconvLayer(
    filters,
    kernel_size=(5, 5),
    padding='same',
    strides=(1, 1),
    bnorm=True,
    dropout=0.0,
    activation='leaky_relu'):

    def fun(inputs):
        if dropout > 0.0:
            x = Dropout(dropout)(inputs)
        else:
            x = inputs

        kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        bias_init = keras.initializers.Zeros()

        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            kernel_initializer=kernel_init,
                            bias_initializer=bias_init,
                            padding=padding)(x)

        if bnorm:
            x = BatchNormalization()(x)

        if activation == 'leaky_relu':
            x = LeakyReLU(0.1)(x)
        elif activation == 'elu':
            x = ELU()(x)
        else:
            x = Activation(activation)(x)

        return x

    return fun

class VAELossLayer(Layer):
    __name__ = 'vae_loss_layer'

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(VAELossLayer, self).__init__(**kwargs)
        self.beta = beta

    def lossfun(self, x_true, x_pred, z_avg, z_log_var):
        rec_loss = K.mean(K.square(x_true - x_pred))
        kl_loss = K.mean(-0.5 * self.beta * K.sum(1.0 + z_log_var - K.square(z_avg) - K.exp(z_log_var), axis=-1))
        return rec_loss + kl_loss

    def call(self, inputs):
        x_true = inputs[0]
        x_pred = inputs[1]
        z_avg = inputs[2]
        z_log_var = inputs[3]
        loss = self.lossfun(x_true, x_pred, z_avg, z_log_var)
        self.add_loss(loss, inputs=inputs)

        return x_true

class ClassifierLossLayer(Layer):
    __name__ = 'classifier_loss_layer'

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(ClassifierLossLayer, self).__init__(**kwargs)

    def lossfun(self, c_true, c_pred):
        return K.mean(keras.metrics.categorical_crossentropy(c_true, c_pred))

    def call(self, inputs):
        c_true = inputs[0]
        c_pred = inputs[1]
        loss = self.lossfun(c_true, c_pred)
        self.add_loss(loss, inputs=inputs)

        return c_true

class DiscriminatorLossLayer(Layer):
    __name__ = 'discriminator_loss_layer'

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(DiscriminatorLossLayer, self).__init__(**kwargs)

    def lossfun(self, y_real, y_fake_f, y_fake_p):
        y_pos = K.ones_like(y_real)
        y_neg = K.zeros_like(y_real)
        loss_real = keras.metrics.binary_crossentropy(y_pos, y_real)
        loss_fake_f = keras.metrics.binary_crossentropy(y_neg, y_fake_f)
        loss_fake_p = keras.metrics.binary_crossentropy(y_neg, y_fake_p)
        return K.mean(loss_real + loss_fake_f + loss_fake_p)

    def call(self, inputs):
        y_real = inputs[0]
        y_fake_f = inputs[1]
        y_fake_p = inputs[2]
        loss = self.lossfun(y_real, y_fake_f, y_fake_p)
        self.add_loss(loss, inputs=inputs)

        return y_real

class GeneratorLossLayer(Layer):
    __name__ = 'generator_loss_layer'

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(GeneratorLossLayer, self).__init__(**kwargs)

    def lossfun(self, x_r, x_f, f_D_x_f, f_D_x_r, f_C_x_r, f_C_x_f):
        loss_x = K.mean(K.square(x_r - x_f))
        loss_d = K.mean(K.square(f_D_x_r - f_D_x_f))
        loss_c = K.mean(K.square(f_C_x_r - f_C_x_f))

        return loss_x + loss_d + loss_c

    def call(self, inputs):
        x_r = inputs[0]
        x_f = inputs[1]
        f_D_x_r = inputs[2]
        f_D_x_f = inputs[3]
        f_C_x_r = inputs[4]
        f_C_x_f = inputs[5]
        loss = self.lossfun(x_r, x_f, f_D_x_r, f_D_x_f, f_C_x_r, f_C_x_f)
        self.add_loss(loss, inputs=inputs)

        return x_r

class FeatureMatchingLayer(Layer):
    __name__ = 'feature_matching_layer'

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(FeatureMatchingLayer, self).__init__(**kwargs)

    def lossfun(self, f1, f2):
        f1_avg = K.mean(f1, axis=0)
        f2_avg = K.mean(f2, axis=0)
        return 0.5 * K.mean(K.square(f1_avg - f2_avg))

    def call(self, inputs):
        f1 = inputs[0]
        f2 = inputs[1]
        loss = self.lossfun(f1, f2)
        self.add_loss(loss, inputs=inputs)

        return f1

class KLLossLayer(Layer):
    __name__ = 'kl_loss_layer'

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(KLLossLayer, self).__init__(**kwargs)

    def lossfun(self, z_avg, z_log_var):
        kl_loss = -0.5 * K.mean(1.0 + z_log_var - K.square(z_avg) - K.exp(z_log_var))
        return kl_loss

    def call(self, inputs):
        z_avg = inputs[0]
        z_log_var = inputs[1]
        loss = self.lossfun(z_avg, z_log_var)
        self.add_loss(loss, inputs=inputs)

        return z_avg

def discriminator_accuracy(x_r, x_f, x_p):
    def accfun(y0, y1):
        x_pos = K.ones_like(x_r)
        x_neg = K.zeros_like(x_r)
        loss_r = K.mean(keras.metrics.binary_accuracy(x_pos, x_r))
        loss_f = K.mean(keras.metrics.binary_accuracy(x_neg, x_f))
        loss_p = K.mean(keras.metrics.binary_accuracy(x_neg, x_p))
        return (1.0 / 3.0) * (loss_r + loss_p + loss_f)

    return accfun

def generator_accuracy(x_p, x_f):
    def accfun(y0, y1):
        x_pos = K.ones_like(x_p)
        loss_p = K.mean(keras.metrics.binary_accuracy(x_pos, x_p))
        loss_f = K.mean(keras.metrics.binary_accuracy(x_pos, x_f))
        return 0.5 * (loss_p + loss_f)

    return accfun

def time_format(t):
    m, s = divmod(t, 60)
    m = int(m)
    s = int(s)
    if m == 0:
        return '%d sec' % s
    else:
        return '%d min %d sec' % (m, s)

class CVAEGAN:
    def __init__(self,
        input_shape=(64, 64, 3),
        num_attrs=40,
        z_dims = 128,
        name='cvaegan',
        **kwargs
    ):
        self.name = name
        self.check_input_shape(input_shape)
        self.input_shape = input_shape

        if 'output' not in kwargs:
            self.output = 'output'
        else:
            self.output = kwargs['output']

        self.test_mode = False
        self.trainers = {}

        self.attr_names = None
        self.input_shape = input_shape
        self.num_attrs = num_attrs
        self.z_dims = z_dims

        self.f_enc = None
        self.f_dec = None
        self.f_dis = None
        self.f_cls = None
        self.enc_trainer = None
        self.dec_trainer = None
        self.dis_trainer = None
        self.cls_trainer = None

        self.build_model()

    def check_input_shape(self, input_shape):
        # Check for CelebA
        if input_shape == (64, 64, 3):
            return

        # Check for MNIST (size modified)
        if input_shape == (32, 32, 1):
            return

        # Check for Cifar10, 100 etc
        if input_shape == (32, 32, 3):
            return

        errmsg = 'Input size should be 32 x 32 or 64 x 64!'
        raise Exception(errmsg)

    def main_loop(self, datasets, samples, attr_names, epochs=100, batchsize=100, reporter=[]):
        self.attr_names = attr_names

        # Create output directories if not exist
        out_dir = os.path.join(self.output, self.name)
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        res_out_dir = os.path.join(out_dir, 'results')
        if not os.path.isdir(res_out_dir):
            os.mkdir(res_out_dir)

        wgt_out_dir = os.path.join(out_dir, 'weights')
        if not os.path.isdir(wgt_out_dir):
            os.mkdir(wgt_out_dir)

        # Start training
        print('\n\n--- START TRAINING ---\n')
        num_data = len(datasets)
        for e in range(epochs):
            perm = np.random.permutation(num_data)
            start_time = time.time()
            for b in range(0, num_data, batchsize):
                bsize = min(batchsize, num_data - b)
                indx = perm[b:b+bsize]

                # Get batch and train on it
                x_batch = self.make_batch(datasets, indx)
                losses = self.train_on_batch(x_batch)

                # Print current status
                ratio = 100.0 * (b + bsize) / num_data
                print(chr(27) + "[2K", end='')
                print('\rEpoch #%d | %d / %d (%6.2f %%) ' % \
                      (e + 1, b + bsize, num_data, ratio), end='')

                for k in reporter:
                    if k in losses:
                        print('| %s = %8.6f ' % (k, losses[k]), end='')

                # Compute ETA
                elapsed_time = time.time() - start_time
                eta = elapsed_time / (b + bsize) * (num_data - (b + bsize))
                print('| ETA: %s ' % time_format(eta), end='')

                sys.stdout.flush()

                # Save generated images
                if (b + bsize) % 10000 == 0 or (b+ bsize) == num_data:
                    outfile = os.path.join(res_out_dir, 'epoch_%04d_batch_%d.png' % (e + 1, b + bsize))
                    self.save_images(samples, outfile)

                if self.test_mode:
                    print('\nFinish testing: %s' % self.name)
                    return

            print('')

            # Save current weights
            self.save_model(wgt_out_dir, e + 1)

    def make_batch(self, datasets, indx):
        '''
        Get batch from datasets
        '''
        return datasets[indx]

    def save_images(self, samples, filename):
        '''
        Save images generated from random sample numbers
        '''
        imgs = self.predict(samples) * 0.5 + 0.5
        imgs = np.clip(imgs, 0.0, 1.0)
        if imgs.shape[3] == 1:
            imgs = np.squeeze(imgs, axis=(3,))

        fig = plt.figure(figsize=(8, 8))
        grid = gridspec.GridSpec(10, 10, wspace=0.1, hspace=0.1)
        for i in range(100):
            ax = plt.Subplot(fig, grid[i])
            if imgs.ndim == 4:
                ax.imshow(imgs[i, :, :, :], interpolation='none', vmin=0.0, vmax=1.0)
            else:
                ax.imshow(imgs[i, :, :], cmap='gray', interpolation='none', vmin=0.0, vmax=1.0)
            ax.axis('off')
            fig.add_subplot(ax)

        fig.savefig(filename, dpi=200)
        plt.close(fig)

    def save_model(self, out_dir, epoch):
        folder = os.path.join(out_dir, 'epoch_%05d' % epoch)
        if not os.path.isdir(folder):
            os.mkdir(folder)

        for k, v in self.trainers.items():
            filename = os.path.join(folder, '%s.hdf5' % (k))
            v.save_weights(filename)

        x_r = datasets.x_test
        c = self.f_cls.predict(x_r)
        z_test = self.f_enc.predict([ x_r, c[0] ])

        filename = os.path.join(folder, 'cvaegan_mnist_activations-%d.csv' % (beta))
        np.savetxt(filename, np.array(z_test), delimiter=",", fmt='%f')


    def store_to_save(self, name):
        self.trainers[name] = getattr(self, name)

    def load_model(self, folder):
        for k, v in self.trainers.items():
            filename = os.path.join(folder, '%s.hdf5' % (k))
            getattr(self, k).load_weights(filename)

    def make_batch(self, datasets, indx):
        images = datasets.images[indx]
        attrs = datasets.attrs[indx]
        return images, attrs

    def save_images(self, samples, filename):
        assert self.attr_names is not None

        num_samples = len(samples)
        attrs = np.identity(self.num_attrs)
        attrs = np.tile(attrs, (num_samples, 1))

        samples = np.tile(samples, (1, self.num_attrs))
        samples = samples.reshape((num_samples * self.num_attrs, -1))

        imgs = 0.5 - self.predict([samples, attrs])
        imgs = np.clip(imgs, 0.0, 1.0)
        if imgs.shape[3] == 1:
            imgs = np.squeeze(imgs, axis=(3,))

        fig = plt.figure(figsize=(self.num_attrs, 10))
        grid = gridspec.GridSpec(num_samples, self.num_attrs, wspace=0.1, hspace=0.1)
        for i in range(num_samples * self.num_attrs):
            ax = plt.Subplot(fig, grid[i])
            if imgs.ndim == 4:
                ax.imshow(imgs[i, :, :, :], interpolation='none', vmin=0.0, vmax=1.0)
            else:
                ax.imshow(imgs[i, :, :], cmap='gray', interpolation='none', vmin=0.0, vmax=1.0)
            ax.axis('off')
            fig.add_subplot(ax)

        fig.savefig(filename, dpi=200)
        plt.close(fig)

    def train_on_batch(self, x_batch):
        x_r, c = x_batch

        batchsize = len(x_r)
        z_p = np.random.normal(size=(batchsize, self.z_dims)).astype('float32')

        x_dummy = np.zeros(x_r.shape, dtype='float32')
        c_dummy = np.zeros(c.shape, dtype='float32')
        z_dummy = np.zeros(z_p.shape, dtype='float32')
        y_dummy = np.zeros((batchsize, 1), dtype='float32')
        f_dummy = np.zeros((batchsize, 8192), dtype='float32')

        # Train autoencoder
        self.enc_trainer.train_on_batch([x_r, c, z_p], [x_dummy, z_dummy])

        # Train generator
        g_loss, _, _, _, _, _, g_acc = self.dec_trainer.train_on_batch([x_r, c, z_p], [x_dummy, f_dummy, f_dummy])

        # Train classifier
        self.cls_trainer.train_on_batch([x_r, c], c_dummy)

        # Train discriminator
        d_loss, d_acc = self.dis_trainer.train_on_batch([x_r, c, z_p], y_dummy)

        loss = {
            'g_loss': g_loss,
            'd_loss': d_loss,
            'g_acc': g_acc,
            'd_acc': d_acc
        }
        return loss

    def predict(self, z_samples):
        return self.f_dec.predict(z_samples)

    def build_model(self):
        self.f_enc = self.build_encoder(output_dims=self.z_dims * 2)
        self.f_dec = self.build_decoder()
        self.f_dis = self.build_discriminator()
        self.f_cls = self.build_classifier()

        # Algorithm
        x_r = Input(shape=self.input_shape)
        c = Input(shape=(self.num_attrs,))
        z_params = self.f_enc([x_r, c])

        z_avg = Lambda(lambda x: x[:, :self.z_dims], output_shape=(self.z_dims,))(z_params)
        z_log_var = Lambda(lambda x: x[:, self.z_dims:], output_shape=(self.z_dims,))(z_params)
        z = Lambda(sample_normal, output_shape=(self.z_dims,))([z_avg, z_log_var])

        kl_loss = KLLossLayer(name="latent")([z_avg, z_log_var])

        z_p = Input(shape=(self.z_dims,))

        x_f = self.f_dec([z, c])
        x_p = self.f_dec([z_p, c])

        y_r, f_D_x_r = self.f_dis(x_r)
        y_f, f_D_x_f = self.f_dis(x_f)
        y_p, f_D_x_p = self.f_dis(x_p)

        d_loss = DiscriminatorLossLayer()([y_r, y_f, y_p])

        c_r, f_C_x_r = self.f_cls(x_r)
        c_f, f_C_x_f = self.f_cls(x_f)
        c_p, f_C_x_p = self.f_cls(x_p)

        g_loss = GeneratorLossLayer()([x_r, x_f, f_D_x_r, f_D_x_f, f_C_x_r, f_C_x_f])
        gd_loss = FeatureMatchingLayer()([f_D_x_r, f_D_x_p])
        gc_loss = FeatureMatchingLayer()([f_C_x_r, f_C_x_p])

        c_loss = ClassifierLossLayer()([c, c_r])

        # Build classifier trainer
        set_trainable(self.f_enc, False)
        set_trainable(self.f_dec, False)
        set_trainable(self.f_dis, False)
        set_trainable(self.f_cls, True)

        self.cls_trainer = Model(inputs=[x_r, c],
                                 outputs=[c_loss])
        self.cls_trainer.compile(loss=[zero_loss],
                                 optimizer=Adam(lr=2.0e-4, beta_1=0.5))
        self.cls_trainer.summary()

        # Build discriminator trainer
        set_trainable(self.f_enc, False)
        set_trainable(self.f_dec, False)
        set_trainable(self.f_dis, True)
        set_trainable(self.f_cls, False)

        self.dis_trainer = Model(inputs=[x_r, c, z_p],
                                 outputs=[d_loss])
        self.dis_trainer.compile(loss=[zero_loss],
                                 optimizer=Adam(lr=2.0e-4, beta_1=0.5),
                                 metrics=[discriminator_accuracy(y_r, y_f, y_p)])
        self.dis_trainer.summary()

        # Build generator trainer
        set_trainable(self.f_enc, False)
        set_trainable(self.f_dec, True)
        set_trainable(self.f_dis, False)
        set_trainable(self.f_cls, False)

        self.dec_trainer = Model(inputs=[x_r, c, z_p],
                                 outputs=[g_loss, gd_loss, gc_loss])
        self.dec_trainer.compile(loss=[zero_loss, zero_loss, zero_loss],
                                 optimizer=Adam(lr=2.0e-4, beta_1=0.5),
                                 metrics=[generator_accuracy(y_p, y_f)])

        # Build autoencoder
        set_trainable(self.f_enc, True)
        set_trainable(self.f_dec, False)
        set_trainable(self.f_dis, False)
        set_trainable(self.f_cls, False)

        self.enc_trainer = Model(inputs=[x_r, c, z_p],
                                outputs=[g_loss, kl_loss])
        self.enc_trainer.compile(loss=[zero_loss, zero_loss],
                                optimizer=Adam(lr=2.0e-4, beta_1=0.5))
        self.enc_trainer.summary()

        # Store trainers
        self.store_to_save('cls_trainer')
        self.store_to_save('dis_trainer')
        self.store_to_save('dec_trainer')
        self.store_to_save('enc_trainer')

    def build_encoder(self, output_dims):
        x_inputs = Input(shape=self.input_shape)
        c_inputs = Input(shape=(self.num_attrs,))

        c = Reshape((1, 1, self.num_attrs))(c_inputs)
        c = UpSampling2D(size=self.input_shape[:2])(c)
        x = Concatenate(axis=-1)([x_inputs, c])

        x = BasicConvLayer(filters=128, strides=(2, 2))(x)
        x = BasicConvLayer(filters=256, strides=(2, 2))(x)
        x = BasicConvLayer(filters=256, strides=(2, 2))(x)
        x = BasicConvLayer(filters=512, strides=(2, 2))(x)

        x = Flatten()(x)
        x = Dense(intermediate_dim)(x)
        x = Activation('relu')(x)

        x = Dense(output_dims)(x)
        x = Activation('linear')(x)

        return Model([x_inputs, c_inputs], x)

    def build_decoder(self):
        z_inputs = Input(shape=(self.z_dims,))
        c_inputs = Input(shape=(self.num_attrs,))
        z = Concatenate()([z_inputs, c_inputs])

        w = self.input_shape[0] // (2 ** 4)
        x = Dense(w * w * 512)(z)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Reshape((w, w, 512))(x)

        x = BasicDeconvLayer(filters=512, strides=(2, 2))(x)
        x = BasicDeconvLayer(filters=256, strides=(2, 2))(x)
        x = BasicDeconvLayer(filters=256, strides=(2, 2))(x)
        x = BasicDeconvLayer(filters=128, strides=(2, 2))(x)

        d = self.input_shape[2]
        x = BasicDeconvLayer(filters=d, strides=(1, 1), bnorm=False, activation='tanh')(x)

        return Model([z_inputs, c_inputs], x)

    def build_discriminator(self):
        inputs = Input(shape=self.input_shape)

        x = BasicConvLayer(filters=128, strides=(2, 2))(inputs)
        x = BasicConvLayer(filters=256, strides=(2, 2))(x)
        x = BasicConvLayer(filters=256, strides=(2, 2))(x)
        x = BasicConvLayer(filters=512, strides=(2, 2))(x)

        f = Flatten()(x)
        x = Dense(intermediate_dim)(f)
        x = Activation('relu')(x)

        x = Dense(1)(x)
        x = Activation('sigmoid')(x)

        return Model(inputs, [x, f])

    def build_classifier(self):
        inputs = Input(shape=self.input_shape)

        x = BasicConvLayer(filters=128, strides=(2, 2))(inputs)
        x = BasicConvLayer(filters=256, strides=(2, 2))(x)
        x = BasicConvLayer(filters=256, strides=(2, 2))(x)
        x = BasicConvLayer(filters=512, strides=(2, 2))(x)

        f = Flatten()(x)
        x = Dense(intermediate_dim)(f)
        x = Activation('relu')(x)

        x = Dense(self.num_attrs)(x)
        x = Activation('softmax')(x)

        return Model(inputs, [x, f])



class Dataset(object):
    def __init__(self):
        self.images = None

    def __len__(self):
        return len(self.images)

    def _get_shape(self):
        return self.images.shape

    shape = property(_get_shape)

class ConditionalDataset(Dataset):
    def __init__(self):
        super(ConditionalDataset, self).__init__()
        self.attrs = None
        self.attr_names = None

def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2)), 'constant', constant_values=0)
    x_train = (x_train[:, :, :, np.newaxis] / 255.0).astype('float32')
    #x_train = x_train[1:10000,:,:,:]
    x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2)), 'constant', constant_values=0)
    x_test = (x_test[:, :, :, np.newaxis] / 255.0).astype('float32')
    y_train = keras.utils.to_categorical(y_train)
    y_train = y_train.astype('float32')

    datasets = ConditionalDataset()
    datasets.images = x_train
    datasets.attrs = y_train
    datasets.x_test = x_test
    datasets.attr_names = [str(i) for i in range(10)]

    return datasets


# Make output direcotiry if not exists
if not os.path.isdir(output):
    os.mkdir(output)

datasets = load_data()

model = CVAEGAN(
    input_shape=datasets.images.shape[1:],
    num_attrs=len(datasets.attr_names),
    z_dims=latent_dim,
    output=output
)

# Training loop
datasets.images = datasets.images * 2.0 - 1.0
samples = np.random.normal(size=(10, latent_dim)).astype(np.float32)
model.main_loop(datasets, samples, datasets.attr_names,
    epochs=epochs,
    batchsize=batch_size,
    reporter=['loss', 'g_loss', 'd_loss', 'g_acc', 'd_acc', 'c_loss', 'ae_loss'])  