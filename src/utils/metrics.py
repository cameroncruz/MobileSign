import tensorflow as tf


def dense_to_sparse(dense, dtype=tf.int64):
    zero = tf.constant(0, dtype=dtype)
    where = tf.not_equal(dense, zero)
    indices = tf.where(where)
    values = tf.gather_nd(dense, indices)

    if tf.executing_eagerly():
        sparse = tf.SparseTensor(indices, values, dense.shape)
    else:
        sparse = tf.SparseTensor(indices, values, tf.shape(dense))

    return sparse


class WER(tf.keras.metrics.Mean):
    def __init__(self):
        super(WER, self).__init__(name='word_error_rate')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_pred_sparse = dense_to_sparse(y_pred)
        y_true = tf.cast(y_true, dtype=tf.int64)
        y_true_sparse = dense_to_sparse(y_true)

        return super(WER, self).update_state(tf.edit_distance(y_pred_sparse, y_true_sparse, normalize=True))
