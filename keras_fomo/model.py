"""Keras/TensorFlow FOMO model.

A MobileNetV2 backbone (ImageNet-pretrained on RGB) followed by a 1x1 conv
head + sigmoid producing a per-cell person-presence probability map.

Grayscale inputs are tiled to 3 channels at the model boundary so we can keep
ImageNet weights without modification.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_fomo_keras(
    *,
    image_size: int = 192,
    grid_size: int = 6,
    backbone: str = "mobilenetv2",
    alpha: float = 1.0,
    pretrained: bool = True,
) -> keras.Model:
    """Construct a FOMO-style Keras model.

    Expects 3-channel inputs already scaled to [-1, 1] (e.g. via
    ``tf.keras.applications.mobilenet_v2.preprocess_input``).
    The backbone is truncated at the MobileNetV2 stride-32 feature map so the
    spatial output for a 192x192 input is naturally 6x6. For other grid sizes
    a ``Resizing`` layer bilinearly resizes the feature map.
    """

    if backbone != "mobilenetv2":
        raise ValueError(f"Unsupported backbone: {backbone}")

    base = keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet" if pretrained else None,
        alpha=alpha,
    )

    inputs = keras.Input(shape=(image_size, image_size, 3), name="image")
    features = base(inputs, training=False)  # (B, image_size/32, image_size/32, 1280*alpha)

    if features.shape[1] != grid_size or features.shape[2] != grid_size:
        features = layers.Resizing(
            grid_size, grid_size, interpolation="bilinear", name="resize_to_grid"
        )(features)

    head = layers.Conv2D(1, kernel_size=1, activation="sigmoid", name="presence")(features)
    model = keras.Model(inputs, head, name="fomo_mobilenetv2")
    return model
