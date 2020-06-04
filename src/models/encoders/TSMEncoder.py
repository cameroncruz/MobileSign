import tensorflow as tf
from models.encoders.TSM import TSMLayer
from utils.mobilenetv2_splitter import get_submodels_mobilenetv2


class TSMEncoder(tf.keras.Model):
    def __init__(self, window_size, temporal_stride, weights_path):
        super(TSMEncoder, self).__init__(dynamic=True)
        self.window_size = window_size
        self.temporal_stride = temporal_stride

        # TODO: For some reason gradients aren't being calculated for these during training?
        base, block_13_14, block_15, block_16, block_final = get_submodels_mobilenetv2(
            weights=weights_path
        )

        self.base = tf.keras.layers.TimeDistributed(base)

        self.block_13_14 = tf.keras.layers.TimeDistributed(block_13_14)
        self.tsm_13_14 = TSMLayer()
        self.downsample_13_14 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(filters=160, kernel_size=1, strides=2)
        )
        self.add_13_14 = tf.keras.layers.Add()

        self.block_15 = tf.keras.layers.TimeDistributed(block_15)
        self.tsm_15 = TSMLayer()
        self.add_15 = tf.keras.layers.Add()

        self.block_16 = tf.keras.layers.TimeDistributed(block_16)
        self.tsm_16 = TSMLayer()
        self.increase_dims_16 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(filters=320, kernel_size=1, strides=1)
        )
        self.add_16 = tf.keras.layers.Add()

        self.block_final = tf.keras.layers.TimeDistributed(block_final)
        self.tsm_final = TSMLayer()

        self.global_pooling = tf.keras.layers.GlobalAveragePooling3D()

    def embed_clip(self, inputs):
        x = self.base(inputs)
        tsm = self.tsm_13_14(x)
        block = self.block_13_14(tsm)
        res = self.downsample_13_14(x)
        x = self.add_13_14([block, res])
        # x = self.add_13_14([self.block_13_14(self.tsm_13_14(x)), self.downsample_13_14(x)])
        x = self.add_15([self.block_15(self.tsm_15(x)), x])
        x = self.add_16([self.block_16(self.tsm_16(x)), self.increase_dims_16(x)])
        x = self.block_final(self.tsm_final(x))
        x = self.global_pooling(x)
        return x

    def call(self, inputs, training=None, mask=None):
        batch_size, n_frames, H, W, C = inputs.shape

        if n_frames < self.window_size:
            inputs = tf.pad(
                inputs,
                tf.constant(
                    [[0, 0], [0, self.window_size - n_frames], [0, 0], [0, 0], [0, 0]]
                ),
            )

        clip_embeddings = [
            self.embed_clip(inputs[:, i : i + self.window_size, :, :, :])
            for i in range(
                0, max(n_frames - self.window_size, 0) + 1, self.temporal_stride
            )
        ]

        clip_embeddings = tf.stack(clip_embeddings, axis=1)

        return clip_embeddings
