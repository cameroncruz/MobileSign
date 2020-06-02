import tensorflow as tf
from models.encoders.TSM import TSMLayer


class TSMEncoder(tf.keras.Model):
    def __init__(self):
        super(TSMEncoder, self).__init__()

    def call(self, inputs, training=None, mask=None):
        pass
