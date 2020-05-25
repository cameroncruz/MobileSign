import tensorflow as tf
from determined.keras import TFKerasTrial, TFKerasTrialContext
from models.BaselineModel import BaselineModel
from utils.dataloader import create_parse_fn, frame_sampling_fn
from utils.metrics import WER


class PhoenixBaselineTrial(TFKerasTrial):
    def __init__(self, context: TFKerasTrialContext) -> None:
        super(PhoenixBaselineTrial, self).__init__(context)

        self.context = context

    def build_model(self) -> tf.keras.Model:
        model = BaselineModel(vocab_size=self.context.get_hparam("vocab_size"),
                              window_size=self.context.get_hparam("window_size"),
                              temporal_stride=self.context.get_hparam("temporal_stride"),
                              embedding_dim=self.context.get_hparam("embedding_dim"),
                              hidden_size=self.context.get_hparam("hidden_size"))

        model = self.context.wrap_model(model)
        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[WER(), tf.keras.metrics.SparseCategoricalAccuracy()],
        )

        return model

    def build_training_data_loader(self) -> tf.data.Dataset:
        @self.context.experimental.cache_train_dataset("rwth-phoenix-2014-tfdataset", "v1", shuffle=True)
        def make_dataset() -> tf.data.Dataset:
            dataset = tf.data.experimental.CsvDataset(
                filenames=train_csv_path,
                record_defaults=[tf.string, tf.string, tf.string, tf.string],
                field_delim="|",
                header=True,
            )
            dataset = dataset.map(create_parse_fn(features_path, vocab_file))
            return dataset

        train_dataset = make_dataset()

        train_dataset = train_dataset.map(frame_sampling_fn)
        train_dataset = train_dataset.padded_batch(self.context.get_per_slot_batch_size(), padded_shapes=([None, 224, 224, 3], [None]))

        return train_dataset

    def build_validation_data_loader(self) -> tf.data.Dataset:
        @self.context.experimental.cache_train_dataset("rwth-phoenix-2014-tfdataset", "v1")
        def make_dataset() -> tf.data.Dataset:
            dataset = tf.data.experimental.CsvDataset(
                filenames=validate_csv_path,
                record_defaults=[tf.string, tf.string, tf.string, tf.string],
                field_delim="|",
                header=True,
            )
            dataset = dataset.map(create_parse_fn(features_path, vocab_file))
            return dataset

        validation_dataset = make_dataset()

        validation_dataset = validation_dataset.map(frame_sampling_fn)
        validation_dataset = validation_dataset.padded_batch(self.context.get_per_slot_batch_size(), padded_shapes=([None, 224, 224, 3], [None]))

        return validation_dataset
