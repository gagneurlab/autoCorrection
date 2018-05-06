from keras.layers import Input, Dense, Lambda, Multiply
from keras.models import Model
from .losses import NB
from .layers import ConstantDispersionLayer
from keras import losses
import os
from keras import backend as K

class Autoencoder():
    def __init__(self, coder_type, size, encoding_dim, seed=None):
        if seed is not None:
            tf.set_random_seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
            sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
            K.set_session(sess)

        if coder_type not in ['autoencoder', 'encoder', 'decoder']:
            raise ValueError('Given coder_type "' + coder_type + 
                             '"is not recognized. Please use "autoencoder", ' +
                             '"encoder" or "decoder".')
        
        self.size = size
        self.coder_type = coder_type
        self.encoding_dim = encoding_dim
        self.Mean_cutoff = lambda x: K.maximum(x, 1e-5)
        self.ClippedExp = lambda x: K.minimum(K.exp(x), 1e10)
        self.pseudoCountLayer = lambda x: x + 1
        self.Loglayer = lambda x: K.log(x)
        self.Invert = lambda x: K.pow(x, -1)
        self.autoenc_model = self.get_autoencoder()
        self.loss = self.set_loss()
        self.model = self.set_model()


    def get_autoencoder(self):
        self.input_layer = Input(shape=(self.size,), name='inp')
        self.pseudoCount = Lambda(self.pseudoCountLayer, output_shape=(self.size,), name="pseudoCount")(self.input_layer)
        self.sf_layer = Input(shape=(self.size,), name='sf')
        self.normalized = Multiply()([self.pseudoCount, self.sf_layer]) 
        self.logcounts = Lambda(self.Loglayer, output_shape=(self.size,), name="logCounts")(self.normalized)
        encoded = Dense(self.encoding_dim, name='encoder', use_bias=True)(self.logcounts)
        decoded = Dense(self.size, name='decoder', use_bias=True)(encoded)
        mean_scaled = Lambda(self.ClippedExp, output_shape=(self.size,), name="mean_scaled")(decoded)
        inv_sf = Lambda(self.Invert, output_shape=(self.size,), name="inv")(self.sf_layer)
        mean = Multiply()([mean_scaled, inv_sf])
        mean_min = Lambda(self.Mean_cutoff, output_shape=(self.size,), name="mean_min")(mean)
        self.disp = ConstantDispersionLayer(name='dispersion')
        self.output = self.disp(mean_min)
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
        if self.coder_type == 'autoencoder':
            nb = NB(self.model.get_layer('dispersion').theta)
            self.loss = nb.loss
        elif self.coder_type == 'encoder':
            self.loss = losses.mean_squared_error
        elif self.coder_type == 'decoder':
            nb = NB(self.model.get_layer('dispersion').theta)
            self.loss = nb.loss
        return self.loss


    def set_model(self):
        if self.coder_type == 'autoencoder':
            self.model = self.get_autoencoder()
        elif self.coder_type == 'encoder':
            self.model = self.get_encoder()
        elif self.coder_type == 'decoder':
            self.model = self.get_decoder()
        return self.model


