import tensorflow as tf
from utils.metrics import WER, dense_to_sparse


class MetricsTest(tf.test.TestCase):
    def testWER(self):
        a = tf.constant([[0, 1, 2]], dtype=tf.int64)
        b = tf.constant([[[0.8, 0.1, 0.1, 0.1],
                          [0.1, 0.8, 0.1, 0.1],
                          [0.1, 0.1, 0.8, 0.1]]])
        c = tf.constant([[[0.8, 0.1, 0.1, 0.1],
                          [0.1, 0.8, 0.1, 0.1],
                          [0.1, 0.1, 0.1, 0.8]]])
        wer = WER()
        wer.update_state(a, b)
        self.assertEqual(wer.result(), 0.0)
        wer.reset_states()
        wer.update_state(a, c)
        self.assertEqual(wer.result(), 0.5)
