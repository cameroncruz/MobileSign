import tensorflow as tf
from models.decoders.Decoder import Decoder


class DecoderTest(tf.test.TestCase):
    def testExpectedOutputShape(self):
        hidden_size = 1024
        vocab_size = 10
        input_shape = (8, 4, 100)
        expected_output_shape = (8, 4, vocab_size)

        decoder = Decoder(hidden_size=hidden_size, vocab_size=vocab_size)

        output = decoder(tf.ones(input_shape))

        self.assertEqual(output.shape, expected_output_shape)
