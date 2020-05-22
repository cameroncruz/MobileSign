import tensorflow as tf
from typing import Callable


def create_parse_fn(features_path: str) -> Callable:

    def parse_example(id: str, folder: str, signer: str, annotation: str) -> (tf.float32, tf.string):
        frames, _ = tf.py_function(read_frames, [features_path, folder], (tf.float32, tf.int32))

        return frames, annotation

    return parse_example


def read_frames(features_path: str, folder: str) -> (tf.float32, tf.int32):
    frame_files = tf.io.gfile.glob(tf.strings.join([features_path, folder]).numpy().decode('utf-8'))
    num_frames = len(frame_files)

    frames = tf.stack([read_and_resize_img(img_file) for img_file in frame_files], axis=0)

    return frames, num_frames


def read_and_resize_img(filename: str) -> tf.float32:
    img = tf.io.read_file(filename)
    img = tf.io.decode_image(img)
    img = tf.image.resize(img, (224, 224))
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img
