import tensorflow as tf
from typing import Tuple


def get_submodels_mobilenetv2(weights="imagenet"):
    mobilenetv2 = tf.keras.applications.MobileNetV2(include_top=False, weights=weights)

    base = tf.keras.Model(
        inputs=mobilenetv2.input, outputs=mobilenetv2.get_layer("block_12_add").output
    )

    block_13_14 = get_block_as_submodel(
        model=mobilenetv2, start=-38, end=-21, block_id="13-14"
    )
    block_15 = get_block_as_submodel(
        model=mobilenetv2, start=-20, end=-12, block_id="15"
    )
    block_16 = get_block_as_submodel(
        model=mobilenetv2, start=-11, end=-4, block_id="16"
    )
    block_final = get_block_as_submodel(
        model=mobilenetv2, start=-3, end=len(mobilenetv2.layers), block_id="final"
    )

    return base, block_13_14, block_15, block_16, block_final


def get_block_as_submodel(model, start, end, block_id):
    start_layer = model.layers[start]
    block_inputs = tf.keras.layers.Input(
        shape=start_layer.input_shape[1:],
        name="mobilenetv2_block_{}_inputs".format(block_id),
    )

    x = block_inputs
    for layer in model.layers[start:end]:
        x = layer(x)

    return tf.keras.Model(
        inputs=block_inputs, outputs=x, name="mobilenetv2_block_{}".format(block_id)
    )
