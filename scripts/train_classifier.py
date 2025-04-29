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
    model = Sequential([
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
    return model


def train_classifier(data_dir: str, model_output: str, epochs: int = 10, batch_size: int = 32):
    # Ensure output directory exists
    os.makedirs(model_output, exist_ok=True)

    mlflow.set_experiment("ALL_Classifier")
    with mlflow.start_run():
        # Prepare datasets
        train_ds, num_classes = parse_csv(f"{data_dir}/train.csv")
        val_ds, _ = parse_csv(f"{data_dir}/val.csv")
        test_ds, _ = parse_csv(f"{data_dir}/test.csv")

        train_ds = train_ds.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # Build & compile custom CNN
        model = build_custom_cnn(num_classes)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Log params
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "num_classes": num_classes
        })

        # Train
        history = model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds
        )

        # Log metrics
        for metric, values in history.history.items():
            for step, value in enumerate(values):
                mlflow.log_metric(metric, value, step=step)

        # Evaluate
        test_loss, test_acc = model.evaluate(test_ds)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_acc)

        # Save model
        mlflow.keras.log_model(model, "model")
        print("✅ Model logged to MLflow")
