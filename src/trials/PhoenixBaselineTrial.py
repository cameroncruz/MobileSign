from typing import List

import tensorflow as tf
from determined.keras import (TFKerasTensorBoard, TFKerasTrial,
                              TFKerasTrialContext)
from models.BaselineModel import BaselineModel
from utils.dataloader import create_parse_fn, create_frame_sampling_fn
from utils.metrics import WER


class PhoenixBaselineTrial(TFKerasTrial):
    def __init__(self, context: TFKerasTrialContext) -> None:
        super(PhoenixBaselineTrial, self).__init__(context)

        self.context = context
        self.data_config = self.context.get_data_config()

    def build_model(self) -> tf.keras.Model:
        model = BaselineModel(
            vocab_size=self.context.get_hparam("vocab_size"),
            window_size=self.context.get_hparam("window_size"),
            temporal_stride=self.context.get_hparam("temporal_stride"),
            embedding_dim=self.context.get_hparam("embedding_dim"),
            hidden_size=self.context.get_hparam("hidden_size"),
        )

        model = self.context.wrap_model(model)
        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[WER(), tf.keras.metrics.SparseCategoricalAccuracy()],
        )

        return model

    def keras_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        return [
            TFKerasTensorBoard(update_freq="batch", histogram_freq=1)
        ]

    def build_training_data_loader(self) -> tf.data.Dataset:
        # @self.context.experimental.cache_train_dataset(
        #     "rwth-phoenix-2014-tfdataset", "v1", shuffle=True
        # )
        def make_dataset() -> tf.data.Dataset:
            dataset = tf.data.experimental.CsvDataset(
                filenames=self.data_config["train_csv"],
                record_defaults=[tf.string, tf.string, tf.string, tf.string],
                field_delim="|",
                header=True,
            )
            dataset = dataset.map(
                create_parse_fn(
                    self.data_config["features_path"] + "train/",
                    self.data_config["vocab_file"],
                )
            )
            return dataset

        train_dataset = make_dataset()

        train_dataset = train_dataset.map(create_frame_sampling_fn(self.context.get_hparam("frame_sampling_stride")))
        train_dataset = train_dataset.padded_batch(
            self.context.get_per_slot_batch_size(),
            padded_shapes=([None, 224, 224, 3], [None]),
        )

        return self.context.wrap_dataset(train_dataset)

    def build_validation_data_loader(self) -> tf.data.Dataset:
        # @self.context.experimental.cache_validation_dataset(
        #     "rwth-phoenix-2014-tfdataset", "v1"
        # )
        def make_dataset() -> tf.data.Dataset:
            dataset = tf.data.experimental.CsvDataset(
                filenames=self.data_config["validation_csv"],
                record_defaults=[tf.string, tf.string, tf.string, tf.string],
                field_delim="|",
                header=True,
            )
            dataset = dataset.map(
                create_parse_fn(
                    self.data_config["features_path"] + "dev/",
                    self.data_config["vocab_file"],
                )
            )
            return dataset

        validation_dataset = make_dataset()

        validation_dataset = validation_dataset.map(create_frame_sampling_fn(self.context.get_hparam("frame_sampling_stride")))
        validation_dataset = validation_dataset.padded_batch(
            self.context.get_per_slot_batch_size(),
            padded_shapes=([None, 224, 224, 3], [None]),
        )

        return self.context.wrap_dataset(validation_dataset)
