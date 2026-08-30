"""M1: Train MobileNetV2 (transfer learning) on cats vs dogs, log to MLflow."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

IMG_SIZE = (224, 224)

def build_model(lr):
    base = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
    base.trainable = False
    model = tf.keras.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="binary_crossentropy", metrics=["accuracy"])
    return model

def make_dataset(split, batch_size, augment=False):
    ds = tf.keras.utils.image_dataset_from_directory(
        "data/processed/" + split, image_size=IMG_SIZE, batch_size=batch_size,
        label_mode="binary", shuffle=(split == "train"), seed=42)
    rescale = tf.keras.layers.Rescaling(1.0 / 255)
    ds = ds.map(lambda x, y: (rescale(x), y))
    if augment:
        aug = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ])
        ds = ds.map(lambda x, y: (aug(x, training=True), y))
    return ds.prefetch(tf.data.AUTOTUNE)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    mlflow.set_experiment("cats-vs-dogs")
    with mlflow.start_run():
        mlflow.log_params({"epochs": args.epochs, "batch_size": args.batch_size,
                           "lr": args.lr, "base_model": "MobileNetV2"})
        train_ds = make_dataset("train", args.batch_size, augment=True)
        val_ds = make_dataset("val", args.batch_size)
        test_ds = make_dataset("test", args.batch_size)
        model = build_model(args.lr)
        history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)
        for epoch in range(args.epochs):
            for k in history.history:
                mlflow.log_metric(k, history.history[k][epoch], step=epoch)
        test_loss, test_acc = model.evaluate(test_ds)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_acc)
        print("test accuracy:", test_acc)
        y_true = np.concatenate([y.numpy() for _, y in test_ds]).ravel()
        y_prob = model.predict(test_ds).ravel()
        y_pred = (y_prob > 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=["cat", "dog"]).plot()
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.figure()
        plt.plot(history.history["loss"], label="train")
        plt.plot(history.history["val_loss"], label="val")
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
        plt.savefig("loss_curve.png")
        mlflow.log_artifact("loss_curve.png")
        Path("models").mkdir(exist_ok=True)
        model.save("models/model.h5")
        mlflow.log_artifact("models/model.h5")

if __name__ == "__main__":
    main()
