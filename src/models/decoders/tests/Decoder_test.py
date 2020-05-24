import tensorflow as tf
from models.decoders.Decoder import Decoder


class DecoderTest(tf.test.TestCase):
    def testExpectedOutputShape(self):
        hidden_size = 1024
        vocab_size = 10
        embedding_dim = 100
        batch_size = 8
        expected_output_shape = (batch_size, vocab_size)

        decoder = Decoder(
            hidden_size=hidden_size, vocab_size=vocab_size, embedding_dim=embedding_dim
        )

        preds, memory_state, carry_state = decoder(
            [
                tf.ones((batch_size, 1)),
                tf.ones((batch_size, 5, 100)),
                tf.ones((batch_size, hidden_size)),
                tf.ones((batch_size, hidden_size)),
            ]
        )

        self.assertEqual(preds.shape, expected_output_shape)
        self.assertEqual(memory_state.shape, (batch_size, hidden_size))
        self.assertEqual(carry_state.shape, (batch_size, hidden_size))
