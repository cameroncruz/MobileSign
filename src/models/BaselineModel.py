import tensorflow as tf
from encoders.VideoEncoder import VideoEncoder
from encoders.TemporalEncoder import TemporalEncoder
from decoders.Decoder import Decoder


class BaselineModel(tf.keras.Model):
    def __init__(self, window_size, temporal_stride, embedding_dim, hidden_size, vocab_size):
        super(BaselineModel, self).__init__()

        self.video_encoder = VideoEncoder(window_size=window_size, temporal_stride=temporal_stride, embedding_dim=embedding_dim)
        self.temporal_encoder = TemporalEncoder(hidden_size=hidden_size)
        self.decoder = Decoder(hidden_size=hidden_size, vocab_size=vocab_size)

    def call(self, inputs, training=None, mask=None):
        clip_embeddings = self.video_encoder(inputs)
        encoder_output = self.temporal_encoder(clip_embeddings)
        return self.decoder(encoder_output)


if __name__ == "__main__":
    model = BaselineModel(8, 4, 100, 1024, 1000)
    model(tf.zeros((4, 16, 224, 224, 3)))
    model.summary()
