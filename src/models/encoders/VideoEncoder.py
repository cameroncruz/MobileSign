import tensorflow as tf


class VideoEncoder(tf.keras.Model):
    def __init__(self, window_size, temporal_stride, embedding_dim):
        super(VideoEncoder, self).__init__()

        self.window_size = window_size
        self.temporal_stride = temporal_stride
        self.embedding_dim = embedding_dim
        self.frame_encoder = tf.keras.applications.MobileNetV2(
            include_top=False, pooling="avg"
        )
        self.frame_encoder.trainable = False
        self.clip_encoder = tf.keras.layers.TimeDistributed(self.frame_encoder)
        self.query = tf.keras.layers.Dense(units=512, activation=None)
        self.value = tf.keras.layers.Dense(units=512, activation=None)
        self.clip_attention = tf.keras.layers.Attention(use_scale=True, causal=True)
        self.clip_flatten = tf.keras.layers.Flatten()
        self.clip_embedding = tf.keras.layers.Dense(units=self.embedding_dim)

    def embed_clip(self, inputs):
        x = self.clip_encoder(inputs)
        x = self.clip_attention([self.query(x), self.value(x)])
        x = self.clip_embedding(self.clip_flatten(x))
        return x

    def call(self, inputs, training=None, mask=None):
        batch_size, n_frames, H, W, C = inputs.shape
        clip_embeddings = [
            self.embed_clip(inputs[:, i : i + self.window_size, :, :, :])
            for i in range(0, n_frames - self.window_size + 1, self.temporal_stride)
        ]
        clip_embeddings = tf.stack(clip_embeddings, axis=1)

        expected_shape = (
            batch_size,
            (n_frames - self.window_size) // self.temporal_stride + 1,
            self.embedding_dim,
        )

        assert clip_embeddings.shape == expected_shape

        return clip_embeddings
