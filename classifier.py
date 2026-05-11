"""
Image Classifier using Transfer Learning (MobileNetV2)
=======================================================
Fine-tunes MobileNetV2 on any image folder organized as:
  data/
    train/
      class_a/ *.jpg
      class_b/ *.jpg
    val/
      class_a/ *.jpg
      class_b/ *.jpg

Also provides a quick CIFAR-10 demo that trains from scratch.
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Lazy TF import (avoids slow startup if just running --help)
# ─────────────────────────────────────────────────────────────────────────────
def _tf():
    import tensorflow as tf
    return tf


# ─────────────────────────────────────────────────────────────────────────────
# Transfer-learning model
# ─────────────────────────────────────────────────────────────────────────────

def build_transfer_model(num_classes: int, img_size: int = 224, trainable_base: bool = False):
    tf = _tf()
    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = trainable_base

    model = tf.keras.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# CIFAR-10 demo (no external data needed)
# ─────────────────────────────────────────────────────────────────────────────

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def demo_cifar10(epochs: int = 5, save_path: str = "cifar10_model.keras"):
    tf = _tf()
    print("\n── Loading CIFAR-10 dataset ─────────────────────────")
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Normalise & one-hot encode
    X_train = X_train.astype("float32") / 255.0
    X_test  = X_test.astype("float32")  / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test  = tf.keras.utils.to_categorical(y_test,  10)

    # Lightweight CNN for quick demo
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same", input_shape=(32, 32, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5, verbose=1),
        tf.keras.callbacks.EarlyStopping(patience=5, restore_bestWeights=True),
    ]

    print(f"\n── Training for {epochs} epochs ───────────────────────")
    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=64,
        callbacks=callbacks,
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n── Test Accuracy: {acc:.4f} ({acc*100:.2f}%) ─────────────")

    model.save(save_path)
    print(f"Model saved → {save_path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Train on custom folder
# ─────────────────────────────────────────────────────────────────────────────

def train_custom(data_dir: str, img_size: int = 224, epochs: int = 10,
                 batch_size: int = 32, save_path: str = "custom_model.keras"):
    tf = _tf()
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    val_dir   = data_dir / "val"

    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=(img_size, img_size), batch_size=batch_size, label_mode="categorical"
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, image_size=(img_size, img_size), batch_size=batch_size, label_mode="categorical"
    )
    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    # Augment training data
    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.map(lambda x, y: (augment(preprocess(x), training=True), y)).prefetch(AUTOTUNE)
    val_ds   = val_ds.map(lambda x, y: (preprocess(x), y)).prefetch(AUTOTUNE)

    model = build_transfer_model(num_classes, img_size)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(save_path, save_best_only=True, monitor="val_accuracy"),
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.3),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    loss, acc = model.evaluate(val_ds, verbose=0)
    print(f"\nVal Accuracy: {acc*100:.2f}%  |  Model → {save_path}")
    return model, class_names


# ─────────────────────────────────────────────────────────────────────────────
# Single image prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_image(model_path: str, image_path: str, class_names: list, img_size: int = 224):
    tf = _tf()
    model = tf.keras.models.load_model(model_path)
    img = tf.keras.utils.load_img(image_path, target_size=(img_size, img_size))
    arr = tf.keras.utils.img_to_array(img)[np.newaxis, ...]

    if img_size == 224:
        arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    else:
        arr = arr / 255.0

    preds = model.predict(arr, verbose=0)[0]
    top_i = np.argsort(preds)[::-1][:3]
    print(f"\nPredictions for {Path(image_path).name}:")
    for i in top_i:
        print(f"  {class_names[i]:<20} {preds[i]*100:.2f}%")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Image Classifier — Transfer Learning")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("demo",  help="Train & evaluate on CIFAR-10 (no data needed)")

    p_train = sub.add_parser("train", help="Train on a custom image folder")
    p_train.add_argument("data_dir")
    p_train.add_argument("--img-size",   type=int, default=224)
    p_train.add_argument("--epochs",     type=int, default=10)
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.add_argument("--save",       default="custom_model.keras")

    p_pred = sub.add_parser("predict", help="Predict class for an image")
    p_pred.add_argument("model")
    p_pred.add_argument("image")
    p_pred.add_argument("--classes", nargs="+", default=CIFAR10_CLASSES)
    p_pred.add_argument("--img-size", type=int, default=32)

    args = parser.parse_args()

    if args.cmd == "demo" or args.cmd is None:
        demo_cifar10()
    elif args.cmd == "train":
        train_custom(args.data_dir, args.img_size, args.epochs, args.batch_size, args.save)
    elif args.cmd == "predict":
        predict_image(args.model, args.image, args.classes, args.img_size)


if __name__ == "__main__":
    main()
