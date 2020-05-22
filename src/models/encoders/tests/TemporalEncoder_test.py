import tensorflow as tf
from models.encoders.TemporalEncoder import TemporalEncoder


class TemporalEncoderTest(tf.test.TestCase):
    def testExpectedOutputShape(self):
        hidden_size = 1024
        input_shape = (4, 8, 100)
        expected_output_shape = (4, 8, hidden_size * 2)

        temporal_encoder = TemporalEncoder(hidden_size=hidden_size)
        output = temporal_encoder(tf.ones(input_shape))

        self.assertEqual(output.shape, expected_output_shape)
