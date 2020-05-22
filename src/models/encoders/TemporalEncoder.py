import tensorflow as tf


class TemporalEncoder(tf.keras.Model):
    def __init__(self, hidden_size):
        super(TemporalEncoder, self).__init__()

        self.blstm1 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units=hidden_size, return_sequences=True, return_state=True
            )
        )
        self.blstm2 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units=hidden_size, return_sequences=True, return_state=False
            )
        )

    def call(self, inputs, training=None, mask=None):
        outputs, *final_state = self.blstm1(inputs)
        outputs = self.blstm2(outputs, initial_state=final_state)

        return outputs
