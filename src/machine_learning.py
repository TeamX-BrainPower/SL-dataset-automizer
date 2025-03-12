from typing import Optional
import keras
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


class MLModel:
    model: keras.models.Sequential
    labels: LabelEncoder
    Y_test: np.ndarray
    X_test: np.ndarray
    __model_trained: bool

    def __init__(self, num_classes: int) -> None:
        self.labels = LabelEncoder()
        self.__model_trained = False
        self.Y_test = np.array([])
        self.X_test = np.array([])
        self.model = keras.models.Sequential(
            layers=[
                keras.layers.Input((30, 44, 1, 3)),
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
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(num_classes, activation="softmax"),
            ],
            name="NTS Model"
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
        # structure the data correctly
        x = []
        y = []

        for word in data:
            sample_label = word["word"]
            for sample in word["samples"]:
                sample_data = np.empty((30, 44, 3))
                sample_data[:] = np.nan

                for idx, value in enumerate(sample[:30]):
                    pose = value.get("pose", None)
                    hands = value.get("hands", None)

                    if pose:
                        landmarks = pose[0]["landmarks"]
                        right_shoulder = landmarks[12]
                        left_shoulder = landmarks[11]
                        right_elbow = landmarks[14]
                        left_elbow = landmarks[13]
                        sample_data[idx, 0:4] = np.array(
                            [
                                [
                                    right_shoulder["x"],
                                    right_shoulder["y"],
                                    right_shoulder["z"],
                                ],
                                [
                                    left_shoulder["x"],
                                    left_shoulder["y"],
                                    left_shoulder["z"],
                                ],
                                [right_elbow["x"], right_elbow["y"], right_elbow["z"]],
                                [left_elbow["x"], left_elbow["y"], left_elbow["z"]],
                            ]
                        )

                    if hands:
                        for hand in hands:
                            landmarks = hand["landmarks"][1:]
                            points = np.array(
                                [[p["x"], p["y"], p["z"]] for p in landmarks]
                            )
                            if hand["handedness"] == "Right":
                                sample_data[idx, 4:24] = points
                            else:
                                sample_data[idx, 24:] = points
                    pass

                x.append(sample_data)
                y.append(sample_label)

        x = np.array(x)
        x = x.reshape((x.shape[0], 30, 44, 1, 3))
        y = np.array(y)

        y_encoded = np.array(self.labels.fit_transform(y))

        X_train, X_test, Y_train, Y_test = train_test_split(
            x, y_encoded, test_size=test_size, stratify=y_encoded
        )

        X_train, X_val, Y_train, Y_val = train_test_split(
            X_train, Y_train, test_size=test_size, stratify=y_encoded
        )

        self.Y_test = np.array(Y_test)
        self.X_test = np.array(X_test)

        # train model
        self.model.fit(
            X_train,
            Y_train,
            validation_data=(X_val, Y_val),
            epochs=epochs,
            batch_size=batchsize,
        )

        self.__model_trained = True

        return

    def save_model(self, path: str) -> None:
        self.model.export(f"{path}/keras")
        new_model = tf.saved_model.load(path)
        converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]  # pyright: ignore
        tf_model = converter.convert()
        open(f"{path}/model.tflite", "wb").write(tf_model)  # pyright: ignore

        return

    def get_model_stats(self, image_path: Optional[str] = None):
        if not self.__model_trained:
            raise ValueError("Model is not trained yet!")

        Y_pred = self.model.predict(self.X_test)
        y_pred = np.argmax(Y_pred)

        print(classification_report(self.Y_test, y_pred))
        fig, ax = plt.subplots(figsize=(13, 12))
        sns.heatmap(confusion_matrix(self.Y_test, y_pred), ax=ax)
        if not image_path:
            fig.show()
        else:
            fig.savefig(image_path)


if __name__ == "__main__":
    import json

    model = MLModel(2)
    with open("data.json", "r+") as f:
        data = json.load(f)
    model.train_model(data)
