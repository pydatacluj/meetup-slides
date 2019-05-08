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
intermediate_dim = 50
latent_dim = 3
batch_size = 64
epochs = 15
beta = 1
denoising = True

np.random.seed(1234)

def neg_loglik(y_true, y_pred):
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)

class KLDivergenceLayer(Layer):
    # Implementation derived from https://tiao.io/post/tutorial-on-variational-autoencoders-with-a-concise-keras-implementation/
    def __init__(self, beta, *args, **kwargs):
        self.is_placeholder = True
        self.beta = beta
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):
        mu, log_var = inputs
        kl_batch = -beta * .5 * K.sum(1 + log_var - K.square(mu) - K.exp(log_var), axis=-1)
        self.add_loss(K.mean(kl_batch), inputs=inputs)
        return inputs

decoder = Sequential([
    Dense(intermediate_dim, input_dim=latent_dim, activation='relu'),
    Dense(sample_size, activation='sigmoid')
])

x = Input(shape=(sample_size,))
h = Dense(intermediate_dim, activation='relu')(x)

z_mu = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

z_mu, z_log_var = KLDivergenceLayer(beta)([z_mu, z_log_var])
z_sigma = Lambda(lambda t: K.exp(.5*t))(z_log_var)

eps_std = 1.0
eps = Input(tensor=K.random_normal(stddev=eps_std, shape=(K.shape(x)[0], latent_dim)))
z_eps = Multiply()([z_sigma, eps])
z = Add()([z_mu, z_eps])

x_pred = decoder(z)

vae = Model(inputs=[x, eps], outputs=x_pred)
vae.compile(optimizer=optimizer, loss=neg_loglik)

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

vae.fit(x_train_altered, x_train, shuffle=True, epochs=epochs, batch_size=batch_size, validation_data=(x_test, x_test))

encoder = Model(x, z_mu)

z_test = encoder.predict(x_test, batch_size=batch_size)

np.savetxt("outputs/vae_mnist_activations-" + str(beta) + ".csv", z_test, delimiter=",", fmt='%f')
np.savetxt("outputs/vae_mnist_classes-" + str(beta) + ".csv", y_test, delimiter=",", fmt='%f')

mesh_dim = 15

u_grid = np.dstack(np.meshgrid(np.linspace(0.05, 0.95, mesh_dim), np.linspace(0.05, 0.95, mesh_dim)))
z_grid = norm.ppf(u_grid)

for i in [ i / 100.0 for i in range(0, 101) ]:
    input = np.full((mesh_dim * mesh_dim, 3), norm.ppf(i))
    input[:,:-1] = z_grid.reshape(mesh_dim * mesh_dim, 2)
    x_decoded = decoder.predict(input)

    np.savetxt("outputs/vae_mnist_manifold-" + str(beta) + "-" + str(i) + ".csv", x_decoded, delimiter=",", fmt='%f')

    #x_decoded = x_decoded.reshape(mesh_dim, mesh_dim, image_size, image_size)
    #plt.figure(figsize=(10, 10))
    #mesh = np.block(list(map(list, x_decoded)))
    #plt.imshow(mesh, cmap='gray')
    #plt.show()