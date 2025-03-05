import keras
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf


class MLModel:
    model: keras.models.Sequential

    def __init__(self, num_classes: int) -> None:
        self.model = keras.models.Sequential(
            [
                keras.layers.Input((30, 46, 1, 3)),
                keras.layers.ConvLSTM2D(
                    filters=32,
                    kernel_size=(3, 1),
                    padding="same",
                    activation="relu",
                    return_sequences=True,
                ),
                keras.layers.BatchNormalization(),
                keras.layers.ConvLSTM2D(
                    filters=64,
                    kernel_size=(3, 1),
                    padding="same",
                    activation="relu",
                    return_sequences=True,
                ),
                keras.layers.BatchNormalization(),
                keras.layers.ConvLSTM2D(
                    filters=128,
                    kernel_size=(3, 1),
                    padding="same",
                    activation="relu",
                    return_sequences=True,
                ),
                keras.layers.BatchNormalization(),
                keras.layers.Flatten(),
                keras.layers.Dense(256, activation="relu"),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(num_classes, activation="softmax"),
            ]
        )

        self.model.summary()
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
        return

    def train_model(
        self,
        data: list[dict],
        test_size: float = 0.25,
        batchsize: int = 32,
        epochs: int = 50,
    ) -> None:
        # do something with the data

        # labels = [d["word"] for d in data]

        # list with index 0 being X and 1 being Y
        all_data = []

        for word in data:
            sample_label = word["word"]
            for sample in word["samples"]:
                sample_data = np.empty((30, 46, 3))
                sample_data[:] = np.nan

                for idx, value in enumerate(sample):
                    pose = value.get("pose", None)
                    hands = value.get("hands", None)

                    if pose:
                        sample_data[idx, 0:6] = np.array(
                            [[p["x"], p["y"], p["z"]] for p in pose["landmarks"]]
                        )

                    if hands:
                        pass
                    pass
                # get the positions

                all_data.append((sample_data, sample_label))

        all_data = np.array(all_data)

        x = all_data[:, 0]
        y = all_data[:, 1]

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        y_one_hot = keras.utils.to_categorical(y_encoded)

        print("data shape:", x.shape, y_one_hot.shape)

        X_train, X_test, Y_train, Y_test = train_test_split(
            x, y_one_hot, test_size=test_size, stratify=y_encoded
        )

        X_train, X_val, Y_train, Y_val = train_test_split(
            X_train, Y_train, test_size=test_size, stratify=y_encoded
        )

        self.model.fit(
            X_train,
            Y_train,
            epochs=epochs,
            batch_size=batchsize,
            validation_data=(X_val, Y_val),
        )

        return

    def save_model(self, path: str) -> None:
        self.model.export(f"{path}/keras")
        new_model = tf.saved_model.load(path)
        converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]  # pyright: ignore
        tf_model = converter.convert()
        open(f"{path}/model.tflite", "wb").write(tf_model)  # pyright: ignore

        return


if __name__ == "__main__":
    model = MLModel(3)

