# scripts/train_classifier.py

import os
import mlflow
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def parse_csv(csv_path, img_size=(224, 224)):
    df = pd.read_csv(csv_path)
    paths = df["path"].tolist()
    labels = sorted(df["label"].unique())
    label2idx = {l: i for i, l in enumerate(labels)}
    y = [label2idx[l] for l in df["label"]]

    def gen():
        for p, yy in zip(paths, y):
            img = tf.io.read_file(p)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, img_size)
            img = img / 255.0
            yield img, yy

    return tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(*img_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    ), len(labels)

def build_custom_cnn(num_classes, input_shape=(224, 224, 3)):
    return Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

def train_classifier(data_dir: str, model_output: str,
                     epochs: int = 10, batch_size: int = 32):
    # 1) configure MLflow to speak to your tracking server
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("ALL_Classifier")

    # 2) local weights folder
    os.makedirs(model_output, exist_ok=True)

    with mlflow.start_run() as run:
        # 3) build datasets
        train_ds, num_classes = parse_csv(f"{data_dir}/train.csv")
        val_ds, _ = parse_csv(f"{data_dir}/val.csv")
        test_ds, _ = parse_csv(f"{data_dir}/test.csv")

        train_ds = train_ds.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds   = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        test_ds  = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # 4) build & train model
        model = build_custom_cnn(num_classes)
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        history = model.fit(train_ds, epochs=epochs, validation_data=val_ds)

        # 5) log metrics & params
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "num_classes": num_classes,
        })
        for metric, vals in history.history.items():
            for step, v in enumerate(vals):
                mlflow.log_metric(metric, v, step=step)

        # 6) test & log
        test_loss, test_acc = model.evaluate(test_ds)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_acc)

        # 7) save & log the Keras model
        mlflow.keras.log_model(model, "model")

        # 8) also export raw weights locally & as an artifact
        weights_path = os.path.join(model_output, "classifier_weights.h5")
        model.save_weights(weights_path)
        mlflow.log_artifact(weights_path, artifact_path="weights")

        print(f"✅ Model + weights saved at {weights_path}")
