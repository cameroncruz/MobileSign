import tensorflow as tf
from models.decoders.Decoder import Decoder
from models.encoders.TemporalEncoder import TemporalEncoder
from models.encoders.VideoEncoder import VideoEncoder


class BaselineModel(tf.keras.Model):
    def __init__(
        self, vocab_size, window_size=8, temporal_stride=4, embedding_dim=100, hidden_size=512
    ):
        super(BaselineModel, self).__init__()

        self.video_encoder = VideoEncoder(
            window_size=window_size,
            temporal_stride=temporal_stride,
            embedding_dim=embedding_dim,
        )
        self.temporal_encoder = TemporalEncoder(hidden_size=hidden_size)
        self.decoder = Decoder(hidden_size=hidden_size, vocab_size=vocab_size, embedding_dim=embedding_dim)

    # def call(self, inputs, training=None, mask=None):
    #     clip_embeddings = self.video_encoder(inputs)
    #     encoder_output = self.temporal_encoder(clip_embeddings)
    #     return self.decoder(encoder_output)

    def train_step(self, data):
        frame_features, targets = data
        batch_size = targets.shape[0]

        loss = 0
        all_preds = []

        memory_state, carry_state = self.decoder.reset_state(batch_size=batch_size)
        decoder_inputs = tf.expand_dims(targets[:, 0], 1)

        with tf.GradientTape() as tape:
            frame_encodings = self.video_encoder(frame_features)
            clip_encodings = self.temporal_encoder(frame_encodings)

            for i in range(1, targets.shape[1]):
                preds, memory_state, carry_state = self.decoder([decoder_inputs, clip_encodings, memory_state, carry_state])

                loss += self.compiled_loss(targets[:, i], preds)

                decoder_inputs = tf.expand_dims(targets[:, i], 1)
                all_preds.append(preds)

        trainable_variables = self.video_encoder.trainable_variables + self.temporal_encoder.trainable_variables + self.decoder.trainable_variables

        gradients = tape.gradient(loss, trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(targets[:, 1:], tf.stack(all_preds, axis=1))
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
