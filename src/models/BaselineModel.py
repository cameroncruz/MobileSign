import tensorflow as tf


class VideoEncoder(tf.keras.Model):
    def __init__(self, window_size, temporal_stride, embedding_dim):
        super(VideoEncoder, self).__init__()

        self.window_size = window_size
        self.temporal_stride = temporal_stride
        self.embedding_dim = embedding_dim
        self.frame_encoder = tf.keras.applications.MobileNetV2(include_top=False, pooling='avg')
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
        clip_embeddings = [self.embed_clip(inputs[:, i:i + self.window_size, :, :, :]) for i in range(0, n_frames - self.window_size + 1, self.temporal_stride)]
        clip_embeddings = tf.stack(clip_embeddings, axis=1)

        expected_shape = (batch_size, (n_frames - self.window_size) // self.temporal_stride + 1, self.embedding_dim)

        assert clip_embeddings.shape == expected_shape

        return clip_embeddings


class TemporalEncoder(tf.keras.Model):
    def __init__(self, hidden_size):
        super(TemporalEncoder, self).__init__()

        self.blstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=hidden_size, return_sequences=True, return_state=True))
        self.blstm2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=hidden_size, return_sequences=True, return_state=False))

    def call(self, inputs, training=None, mask=None):
        outputs, *final_state = self.blstm1(inputs)
        outputs = self.blstm2(outputs, initial_state=final_state)

        return outputs


class Decoder(tf.keras.Model):
    def __init__(self, hidden_size, vocab_size):
        super(Decoder, self).__init__()
        self.query = tf.keras.layers.Dense(units=512, activation=None)
        self.value = tf.keras.layers.Dense(units=512, activation=None)
        self.attention = tf.keras.layers.Attention(use_scale=True)
        self.lstm = tf.keras.layers.LSTM(units=hidden_size, return_sequences=True)
        self.classifier = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=vocab_size, activation='softmax'))

    def call(self, inputs, training=None, mask=None):
        x = self.attention([self.query(inputs), self.value(inputs)])
        x = self.lstm(x)
        preds = self.classifier(x)
        return preds


class BaselineModel(tf.keras.Model):
    def __init__(self):
        super(BaselineModel, self).__init__()

        self.video_encoder = VideoEncoder(window_size=8, temporal_stride=4, embedding_dim=100)
        self.temporal_encoder = TemporalEncoder(hidden_size=1024)
        self.decoder = Decoder(hidden_size=1024, vocab_size=1000)

    def call(self, inputs, training=None, mask=None):
        clip_embeddings = self.video_encoder(inputs)
        encoder_output = self.temporal_encoder(clip_embeddings)
        return self.decoder(encoder_output)


if __name__ == "__main__":
    model = BaselineModel()
    model(tf.zeros((32, 16, 224, 224, 3)))
