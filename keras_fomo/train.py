"""Train the Keras FOMO model on the overhead-person-detection dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from keras_fomo.data import build_dataset, split_size
from keras_fomo.model import build_fomo_keras


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output-dir", default="keras_fomo/runs/fomo")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--grid-size", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=1.0, help="MobileNetV2 width multiplier.")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--brightness-jitter", type=float, default=0.0)
    parser.add_argument("--contrast-jitter", type=float, default=0.0)
    parser.add_argument("--count-aux-weight", type=float, default=0.0,
                        help="Weight on the L1 count-regression auxiliary loss (0 disables)." )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--int8-rep-samples", type=int, default=128)
    parser.add_argument("--no-tflite", action="store_true")
    return parser.parse_args()


def make_representative_dataset(dataset_root: str, manifest: str | None, image_size: int, grid_size: int, num_samples: int):
    """Yield batches of (1, H, W, 1) float32 inputs for INT8 calibration."""

    def gen():
        ds = build_dataset(
            dataset_root,
            split="train",
            manifest_path=manifest,
            image_size=image_size,
            grid_size=grid_size,
            batch_size=1,
            shuffle=True,
            augment=False,
        )
        seen = 0
        for image, _ in ds:
            yield [tf.cast(image, tf.float32)]
            seen += 1
            if seen >= num_samples:
                break

    return gen


def main() -> None:
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    train_ds = build_dataset(
        args.dataset_root,
        split="train",
        manifest_path=args.manifest,
        image_size=args.image_size,
        grid_size=args.grid_size,
        batch_size=args.batch_size,
        shuffle=True,
        augment=not args.no_augment,
        brightness_jitter=args.brightness_jitter,
        contrast_jitter=args.contrast_jitter,
    )
    val_ds = build_dataset(
        args.dataset_root,
        split="val",
        manifest_path=args.manifest,
        image_size=args.image_size,
        grid_size=args.grid_size,
        batch_size=args.batch_size,
        shuffle=False,
        augment=False,
    )

    n_train = split_size(args.dataset_root, "train", args.manifest)
    n_val = split_size(args.dataset_root, "val", args.manifest)
    steps_per_epoch = max(n_train // args.batch_size, 1)
    val_steps = max(int(np.ceil(n_val / args.batch_size)), 1)

    model = build_fomo_keras(
        image_size=args.image_size,
        grid_size=args.grid_size,
        alpha=args.alpha,
        pretrained=not args.no_pretrained,
    )

    if args.count_aux_weight > 0:
        aux_w = float(args.count_aux_weight)

        def _bce_plus_count(y_true, y_pred):
            bce = tf.reduce_mean(keras.losses.binary_crossentropy(y_true, y_pred))
            pred_counts = tf.reduce_sum(y_pred, axis=[1, 2, 3])
            gt_counts = tf.reduce_sum(y_true, axis=[1, 2, 3])
            count_l1 = tf.reduce_mean(tf.abs(pred_counts - gt_counts))
            return bce + aux_w * count_l1

        loss_fn = _bce_plus_count
    else:
        loss_fn = "binary_crossentropy"

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss=loss_fn,
        metrics=[keras.metrics.BinaryAccuracy(name="acc"), keras.metrics.Precision(name="prec"), keras.metrics.Recall(name="rec")],
    )

    ckpt_path = output_dir / "best.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(str(ckpt_path), monitor="val_loss", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        keras.callbacks.CSVLogger(str(output_dir / "training_log.csv")),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=callbacks,
    )

    final_path = output_dir / "final.keras"
    model.save(final_path)

    if args.no_tflite:
        return

    # TF 2.16 + Python 3.12: tf.saved_model.save crashes on Keras 3 _DictWrapper.
    # Bypass both saved_model and from_keras_model by converting from a
    # concrete function instead.
    @tf.function
    def _serving_fn(inputs):
        return model(inputs, training=False)

    concrete_fn = _serving_fn.get_concrete_function(
        tf.TensorSpec(
            shape=[None, args.image_size, args.image_size, 3],
            dtype=tf.float32,
            name="image",
        )
    )

    # ---- TFLite export (float32) ----
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn], trackable_obj=model)
    float_tflite = converter.convert()
    (output_dir / "model_float32.tflite").write_bytes(float_tflite)

    # ---- TFLite export (INT8) ----
    converter_int8 = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn], trackable_obj=model)
    converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_int8.representative_dataset = make_representative_dataset(
        args.dataset_root, args.manifest, args.image_size, args.grid_size, args.int8_rep_samples
    )
    converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_int8.inference_input_type = tf.int8
    converter_int8.inference_output_type = tf.int8
    try:
        int8_tflite = converter_int8.convert()
        (output_dir / "model_int8.tflite").write_bytes(int8_tflite)
    except Exception as exc:  # pragma: no cover - depends on TF version
        print(f"[tflite] INT8 conversion failed: {exc}")

    print(f"Done. Artefacts in {output_dir}")


if __name__ == "__main__":
    main()
