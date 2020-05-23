import tensorflow as tf
import json
from typing import Callable


def create_parse_fn(features_path: str, vocab_file: str) -> Callable:
    """Create a function to parse dataset examples.

    Args:
        features_path: Path to folders of frames.
        vocab_file: JSON of saved Keras Tokenizer.

    Returns:
        Function that performs the following transformation:
        - expects id, folder, signer, and annotation
        - returns preprocessed frames and tokenized annotation converted to word indices
    """
    with open(vocab_file) as f:
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json.load(f))

    def parse_example(
        id: str, folder: str, signer: str, annotation: str
    ) -> (tf.float32, tf.int32):
        frames, _ = tf.py_function(
            read_frames, [features_path, folder], [tf.float32, tf.int32]
        )
        tokenized_annotation = tf.py_function(
            create_tokenize_fn(tokenizer), [annotation], tf.int32
        )

        return frames, tokenized_annotation

    return parse_example


def read_frames(features_path: str, folder: str) -> (tf.float32, tf.int32):
    """Read video frames from folder of frame images.

    Args:
        features_path: Path to folders of frames.
        folder: Folder containing frames for a specific video.

    Returns:
        Frames stacked temporally as a Tensor and number of frames in the video.
    """
    frame_files = tf.io.gfile.glob(
        tf.strings.join([features_path, folder]).numpy().decode("utf-8")
    )
    num_frames = len(frame_files)

    frames = tf.stack(
        [read_and_resize_img(img_file) for img_file in frame_files], axis=0
    )

    return frames, num_frames


def read_and_resize_img(filename: str, preprocess="mobilenet_v2") -> tf.float32:
    """Read an image file and resize HxW. Optionally, apply preprocessing.

    Args:
        filename: Image file.

    Returns:
        Preprocessed tf.float32 Tensor of shape (224, 224, 3) of given image.
    """
    img = tf.io.read_file(filename)
    img = tf.io.decode_image(img)
    img = tf.image.resize(img, (224, 224))
    if preprocess == "mobilenet_v2":
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img


def create_tokenize_fn(
    tokenizer: tf.keras.preprocessing.text.Tokenizer,
    start_token: str = "<START>",
    end_token: str = "<END>",
) -> Callable:
    """Create a function to tokenize annotations.

    Args:
        tokenizer: Keras tokenizer to use.
        start_token: Start token used in corpus.
        end_token: End token used in corpus.

    Returns:
        Annotation that has been tokenized and converted to a sequence of word indices.
    """

    def tokenize(annotation: str):
        tokenized = tokenizer.texts_to_sequences(
            [
                tf.strings.join([start_token, annotation, end_token], separator=" ")
                .numpy()
                .decode("utf-8")
            ]
        )
        return tokenized

    return tokenize
