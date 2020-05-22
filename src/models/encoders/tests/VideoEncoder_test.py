import tensorflow as tf
from models.encoders.VideoEncoder import VideoEncoder


class VideoEncoderTest(tf.test.TestCase):
    def testExpectedOutputShape(self):
        window_size = 8
        temporal_stride = 4
        embedding_dim = 100
        n_frames = 32
        input_shape = (4, n_frames, 160, 160, 3)
        expected_output_shape = (
            4,
            (n_frames - window_size) // temporal_stride + 1,
            embedding_dim,
        )

        video_encoder = VideoEncoder(
            window_size=window_size,
            temporal_stride=temporal_stride,
            embedding_dim=embedding_dim,
        )

        output = video_encoder(tf.ones(input_shape))

        self.assertEqual(output.shape, expected_output_shape)
