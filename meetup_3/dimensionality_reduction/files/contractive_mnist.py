import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras import backend as K
from keras.layers import Input, Dense, Lambda, Layer, Add, Multiply
from keras.models import Model, Sequential
from keras.datasets import mnist
from keras import regularizers

optimizer = "Nadam"
image_size = 28
sample_size = image_size * image_size
intermediate_dim = 100
latent_dim = 3
batch_size = 64
epochs = 15
denoising = True
contr_lambda = 1e-4

inputs = Input(shape=(sample_size,))
encoded = Dense(intermediate_dim, activation='relu')(inputs)
latent = Dense(latent_dim, activation='relu', name='latent')(encoded)
decoded = Dense(intermediate_dim, activation='relu')(latent)
outputs = Dense(sample_size, activation='linear')(decoded)

ae = Model(input=inputs, output=outputs)

input = Input(shape=(sample_size,))
encoder = Model(input, ae.layers[2](ae.layers[1](ae.layers[0](input))))

def contractive_mse(y_pred, y_true):
    mse = K.mean(K.square(y_true - y_pred), axis=1)
    W = K.variable(value=ae.get_layer('latent').get_weights()[0])
    W = K.transpose(W)
    h = ae.get_layer('latent').get_output_at(0)
    dh = h * (1 - h)
    return mse + contr_lambda * K.sum(dh**2 * K.sum(W**2, axis=1), axis=1)


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, sample_size) / 255.
if not denoising:
    x_train_altered = x_train
else:
    x_train_altered = np.concatenate((
        (x_train + 0.25 * np.random.normal(0, 1, x_train.shape)).clip(0, 1),
        (x_train + 0.5 * np.random.normal(0, 1, x_train.shape)).clip(0, 1),
        (x_train + np.random.normal(0, 1, x_train.shape)).clip(0, 1),
        x_train
    ), axis=0)
    x_train = np.concatenate((
        x_train,
        x_train,
        x_train,
        x_train
    ), axis=0)
x_test = x_test.reshape(-1, sample_size) / 255.

ae.compile(optimizer=optimizer, loss=contractive_mse)
ae.fit(x_train_altered, x_train, shuffle=True, epochs=epochs, batch_size=batch_size, validation_data=(x_test, x_test))

z_test = encoder.predict(x_test, batch_size=batch_size)
np.savetxt("outputs/contractive_mnist_activations.csv", z_test, delimiter=",", fmt='%f')
np.savetxt("outputs/contractive_mnist_classes.csv", y_test, delimiter=",", fmt='%f')
