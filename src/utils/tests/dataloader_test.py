import json
import os

import tensorflow as tf
from utils.dataloader import (create_parse_fn, create_tokenize_fn,
                              read_and_resize_img, read_frames)


class DataloaderTest(tf.test.TestCase):
    def setUp(self):
        self.data_path = "../debug_data/phoenix-2014-multisigner/"
        self.features_path = self.data_path + "features/fullFrame-210x260px/"
        self.vocab_file = "../debug_data/phoenix-tokenizer.json"
        self.id = "01April_2010_Thursday_heute_default-1"
        self.folder = "01April_2010_Thursday_heute_default-1/1/*.png"
        self.signer = "Signer04"
        self.annotation = (
            "ICH OSTERN WETTER ZUFRIEDEN MITTAG TEMPERATUR  SUED WARM MEIN NICHT"
        )
        self.tokenized_annotation = [2, 254, 479, 31, 641, 93, 73, 22, 82, 490, 80, 3]
        self.num_frames = 215

    def testReadFrames(self):
        if not os.path.isdir(self.features_path):
            self.skipTest(reason="Debug data not found.")

        expected_shape = (self.num_frames, 224, 224, 3)
        frames, n_frames = read_frames(
            features_path=self.features_path, folder=self.folder
        )

        self.assertEqual(n_frames, self.num_frames)
        self.assertEqual(frames.shape, expected_shape)

    def testReadAndResizeImg(self):
        if not os.path.isdir(self.features_path):
            self.skipTest(reason="Debug data not found.")

        img_path = (
            self.features_path
            + self.folder[:-5]
            + "01April_2010_Thursday_heute.avi_pid0_fn000000-0.png"
        )
        expected_shape = (224, 224, 3)
        img = read_and_resize_img(img_path)

        self.assertEqual(img.shape, expected_shape)

    def testCreateTokenizeFn(self):
        if not os.path.isfile(self.vocab_file):
            self.skipTest(reason="Debug data not found.")

        with open(self.vocab_file) as f:
            tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json.load(f))

        tokenize_fn = create_tokenize_fn(tokenizer)
        tokenized = tokenize_fn(self.annotation)

        self.assertEqual(tokenized[0], self.tokenized_annotation)

    def testParseExample(self):
        if not os.path.isdir(self.features_path):
            self.skipTest(reason="Debug data not found.")

        parse_example = create_parse_fn(self.features_path, self.vocab_file)
        frames, label = parse_example(
            self.id, self.folder, self.signer, self.annotation
        )

        self.assertEqual(frames.shape, (self.num_frames, 224, 224, 3))
        self.assertAllEqual(label, self.tokenized_annotation)

    def testMakeDataset(self):
        csv_path = self.data_path + "annotations/manual/dev.corpus.csv"
        default_types = [tf.string, tf.string, tf.string, tf.string]

        if not os.path.isfile(csv_path):
            self.skipTest(reason="Debug data not found.")

        dataset = tf.data.experimental.CsvDataset(
            filenames=csv_path,
            record_defaults=default_types,
            field_delim="|",
            header=True,
        )
        dataset = dataset.map(create_parse_fn(self.features_path, self.vocab_file))
