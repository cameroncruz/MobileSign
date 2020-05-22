import tensorflow as tf
import os
from utils.dataloader import create_parse_fn, read_frames, read_and_resize_img


class DataloaderTest(tf.test.TestCase):
    def setUp(self):
        self.data_path = "../debug_data/phoenix-2014-multisigner/"
        self.features_path = self.data_path + "features/fullFrame-210x260px/"
        self.id = "01April_2010_Thursday_heute_default-1"
        self.folder = "01April_2010_Thursday_heute_default-1/1/*.png"
        self.signer = "Signer04"
        self.annotation = "ICH OSTERN WETTER ZUFRIEDEN MITTAG TEMPERATUR  SUED WARM MEIN NICHT"
        self.num_frames = 215

    def testReadFrames(self):
        expected_shape = (self.num_frames, 224, 224, 3)
        frames, n_frames = read_frames(features_path=self.features_path, folder=self.folder)

        self.assertEqual(n_frames, self.num_frames)
        self.assertEqual(frames.shape, expected_shape)

    def testReadAndResizeImg(self):
        img_path = self.features_path + self.folder[:-5] + "01April_2010_Thursday_heute.avi_pid0_fn000000-0.png"
        expected_shape = (224, 224, 3)
        img = read_and_resize_img(img_path)

        self.assertEqual(img.shape, expected_shape)

    def testParseExample(self):
        parse_example = create_parse_fn(self.features_path)
        frames, label = parse_example(self.id, self.folder, self.signer, self.annotation)

        self.assertEqual(frames.shape, (self.num_frames, 224, 224, 3))
        self.assertEqual(label, self.annotation)

    def testMakeDataset(self):
        csv_path = self.data_path + "annotations/manual/dev.corpus.csv"
        default_types = [tf.string, tf.string, tf.string, tf.string]

        if not os.path.isfile(csv_path):
            self.skipTest(reason="Debug csv file not found.")

        dataset = tf.data.experimental.CsvDataset(filenames=csv_path,  record_defaults=default_types, field_delim="|", header=True)
        dataset.map(create_parse_fn(self.features_path))
