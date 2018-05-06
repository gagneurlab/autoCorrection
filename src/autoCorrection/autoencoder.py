import numpy as np
from keras.layers import Input, Dense, Lambda, Multiply
from keras.models import Model
from .losses import NB
from .layers import ConstantDispersionLayer
from keras.optimizers import *
from keras import losses
import os
from keras import backend as K


class Autoencoder():
    def __init__(self, encoding_dim=2, size=10000, optimizer=Adam(lr=0.001),
                 choose_autoencoder=False, choose_encoder=False,
                 choose_decoder=False, epochs=1100, batch_size=None, seed=None):
        if seed is not None:
            np.random.seed(seed)
            tf.set_random_seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            tf.reset_default_graph()
            session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
            sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
            K.set_session(sess)
        self.encoding_dim = encoding_dim
        self.size = size
        self.epochs = epochs
        self.batch_size = batch_size
        self.ClippedExp = lambda x: K.minimum(K.exp(x), 1e5)
        self.Invert = lambda x: K.pow(x, -1)
        self.choose_autoencoder = choose_autoencoder
        self.choose_encoder = choose_encoder
        self.choose_decoder = choose_decoder
        self.autoenc_model = self.get_autoencoder()
        self.optimizer = optimizer
        self.loss = self.set_loss()
        self.model = self.set_model()


    def get_autoencoder(self):
        self.input_layer = Input(shape=(self.size,), name='inp')
        self.sf_layer = Input(shape=(self.size,), name='sf')
        self.normalized = Multiply()([self.input_layer, self.sf_layer]) #scale factor layer contains inversed sf see: data_utils.py, TrainTestPreparation
        encoded = Dense(self.encoding_dim, name='encoder', use_bias=True)(self.normalized)
        decoded = Dense(self.size, name='decoder', use_bias=True)(encoded)
        mean_scaled = Lambda(self.ClippedExp, output_shape=(self.size,), name="mean_scaled")(decoded)
        inv_sf = Lambda(self.Invert, output_shape=(self.size,), name="inv")(self.sf_layer)
        mean = Multiply()([mean_scaled, inv_sf])
        self.disp = ConstantDispersionLayer(name='dispersion')
        self.output = self.disp(mean)
        self.model = Model([self.input_layer, self.sf_layer], self.output)
        return self.model

    def get_encoder(self):
        self.encoder = Model([self.autoenc_model.get_layer('inp').input,
                              self.autoenc_model.get_layer('sf').input],
                             self.autoenc_model.get_layer('encoder').output)
        return self.encoder

    def get_decoder(self):
        encoded_input = Input(shape=(self.encoding_dim,))
        decoder_layer = self.autoenc_model.get_layer('decoder')
        decoded = decoder_layer(encoded_input)
        mean_layer = self.autoenc_model.get_layer('mean')
        mean = mean_layer(decoded)
        dispersion_layer = ConstantDispersionLayer(name='dispersion')
        output = dispersion_layer(mean)
        self.decoder = Model(encoded_input, output)
        return self.decoder

    def set_loss(self):
        if self.choose_autoencoder:
            nb = NB(self.model.get_layer('dispersion').theta)
            self.loss = nb.loss
        elif self.choose_encoder:
            self.loss = losses.mean_squared_error
        elif self.choose_decoder:
            nb = NB(self.model.get_layer('dispersion').theta)
            self.loss = nb.loss
        return self.loss

    def set_model(self):
        if self.choose_autoencoder:
            self.model = self.get_autoencoder()
        elif self.choose_encoder:
            self.model = self.get_encoder()
        elif self.choose_decoder:
            self.model = self.get_decoder()
        return self.model




