from keras.optimizers import RMSprop, Adam
from .autoencoder import Autoencoder

class OptimizationModel():
    def __init__(self):
        pass

    def model(self, train_data, lr=0.001,
              encoding_dim=128):
        size = train_data[0]["inp"].shape[1]
        ae = Autoencoder(coder_type="autoencoder", size=size,
                         encoding_dim=encoding_dim)
        ae.model.compile(optimizer=Adam(lr=lr), loss=ae.loss)#metrics=['eval.loss']
        model = ae.model
        return model
