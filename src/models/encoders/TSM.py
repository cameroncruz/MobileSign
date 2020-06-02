import tensorflow as tf


class TSMLayer(tf.keras.layers.Layer):
    def __init__(self, n_segments=8, n_folds=8):
        super(TSMLayer, self).__init__()

        self.n_segments = n_segments
        self.n_folds = n_folds

    def call(self, inputs, **kwargs):
        inputs = tf.transpose(inputs, perm=[0, 1, 4, 2, 3])
        batch_size, n_frames, c, h, w = inputs.shape

        n_batch = n_frames // self.n_segments
        x = tf.reshape(inputs, shape=(batch_size, n_batch, self.n_segments, c, h, w))

        fold = c // self.n_folds

        out = tf.zeros_like(x)
        out[:, :, :-1, :fold] = x[:, :, 1:, :fold]  # shift left
        out[:, :, 1:, fold: 2 * fold] = x[:, :, :-1, fold: 2 * fold]  # shift right
        out[:, :, :, 2 * fold:] = x[:, :, :, 2 * fold:]  # not shift

        out = tf.reshape(out, shape=(batch_size, n_frames, c, h, w))

        return tf.transpose(out, perm=[0, 1, 3, 4, 2])


class TSMModel(tf.keras.Model):
    def __init__(self):
        super(TSMModel, self).__init__()


    def call(self, inputs, training=None, mask=None):
        pass
