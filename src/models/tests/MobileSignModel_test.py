import os

import tensorflow as tf
from models.MobileSignModel import MobileSignModel
from utils.dataloader import create_parse_fn
from utils.metrics import WER


class MobileSignModelTest(tf.test.TestCase):
    def setUp(self):
        self.vocab_file = "../debug_data/phoenix-tokenizer.json"
        self.data_path = "../debug_data/phoenix-2014-multisigner/"
        self.features_path = self.data_path + "features/fullFrame-210x260px/"

    def testTrainMobileSignModel(self):
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

        def slice_fn(frames, label):
            return frames[:32, :, :, :], label

        dataset = dataset.map(slice_fn)
        dataset = dataset.padded_batch(2, padded_shapes=([None, 224, 224, 3], [None]))

        model = MobileSignModel(vocab_size=980, weights_path="mobilenetv2_pretrained.h5")
        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[WER(), tf.keras.metrics.SparseCategoricalAccuracy()],
        )
        tf.saved_model.save(model, export_dir='../debug_data')
        # model.fit(dataset, validation_data=dataset, epochs=100)
