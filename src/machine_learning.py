from typing import Optional
import keras
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

def get_data_from_dict(word: dict) -> tuple[np.ndarray, np.ndarray]:
    sample_label = word["word"]
    x = []
    y = []
    for sample in word["frameData"]:
        sample_data = np.zeros((30, 44, 3))

        for idx, value in enumerate(sample):
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

        x.append(sample_data)
        y.append(sample_label)

    return x, y

def get_predict_data_from_dict(data: list[dict]) -> np.ndarray:
    sample_data = np.zeros((30, 44, 3))

    for idx, value in enumerate(data):
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
    return sample_data

class MLModel:
    model: keras.models.Sequential | None
    labels: LabelEncoder
    x_test: np.ndarray
    y_test: np.ndarray
    __model_trained: bool
    early_stop: keras.callbacks.EarlyStopping
    labels: Optional[list[str]]

    def __init__(self, num_classes: Optional[int] = None, labels: Optional[list[str]] = None) -> None:
        self.labels = LabelEncoder()
        self.__model_trained = False
        self.x_test = np.array([])
        self.y_test = np.array([])
        self.labels = labels
        if not num_classes:
            self.model = None
            print("Model is not initialized as num_classes is None.")
            print("Load a model to initialize it")
            return
        
        self.model: tf.keras.models.Sequential = keras.models.Sequential(
            layers=[
                # keras.layers.Input((30, 44, 1, 3)),
                tf.keras.layers.Input((30, 44, 1, 3, 1)),
                # keras.layers.Reshape((30, 44 * 1 * 3)),
                # keras.layers.LSTM(32, return_sequences=True, activation="relu"),
                # keras.layers.LSTM(64, return_sequences=True, activation="tanh"),
                # keras.layers.LSTM(128, return_sequences=True, activation="tanh"),
                # keras.layers.LSTM(256, return_sequences=True, activation="relu"),
                # keras.layers.LSTM(128, return_sequences=False, activation="tanh"),
                tf.keras.layers.ConvLSTM3D(
                    filters=32,
                    kernel_size=(3, 3, 1),
                    padding="same",
                    activation="relu",
                    return_sequences=True,
                ),
                tf.keras.layers.ConvLSTM3D(
                    filters=64,
                    kernel_size=(3, 3, 1),
                    padding="same",
                    activation="relu",
                    return_sequences=True,
                ),
                tf.keras.layers.ConvLSTM3D(
                    filters=128,
                    kernel_size=(3, 3, 1),
                    padding="same",
                    activation="relu",
                    return_sequences=True,
                ),
                # keras.layers.ConvLSTM2D(
                #     filters=32,
                #     kernel_size=(3, 1),
                #     padding="same",
                #     activation="relu",
                #     return_sequences=True,
                # ),
                # # keras.layers.BatchNormalization(),
                # keras.layers.ConvLSTM2D(
                #     filters=64,
                #     kernel_size=(3, 1),
                #     padding="same",
                #     activation="relu",
                #     return_sequences=True,
                # ),
                # # keras.layers.BatchNormalization(),
                # keras.layers.ConvLSTM2D(
                #     filters=128,
                #     kernel_size=(3, 1),
                #     padding="same",
                #     activation="relu",
                #     return_sequences=True,
                # ),
                # keras.layers.BatchNormalization(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(num_classes, activation="softmax"),
            ],
            name="NTS_Model"
        )

        self.model.summary()
        self.early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=50)
        optimizer = keras.optimizers.Adam(learning_rate=0.0005)
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
        return

    def train_model(
        self,
        train_data: list[dict],
        test_data: list[dict],
        val_data: list[dict],
        batchsize: int = 32,
        epochs: int = 50
    ) -> None:
        # structure the data correctly
        x = []
        y = []

        for word in train_data:
            x_int, y_int = get_data_from_dict(word)
            x.extend(x_int)
            y.extend(y_int)

        x = np.array(x)
        x = x.reshape((x.shape[0], 30, 44, 1, 3, 1))
        y = np.array(y)

        y_encoded = np.array(self.labels.fit_transform(y))

        x_test = []
        y_test = []

        for word in test_data:
            x_test_int, y_test_int = get_data_from_dict(word)
            x_test.extend(x_test_int)
            y_test.extend(y_test_int)

        self.y_test = np.array(self.labels.transform(y_test))
        self.x_test = np.array(x_test)
        self.x_test = self.x_test.reshape((self.x_test.shape[0], 30, 44, 1, 3, 1))

        x_val = []
        y_val = []

        for word in val_data:
            x_val_int, y_val_int = get_data_from_dict(word)
            x_val.extend(x_val_int)
            y_val.extend(y_val_int)

        y_val_encoded = np.array(self.labels.transform(y_val))
        x_val = np.array(x_val)
        x_val = x_val.reshape((x_val.shape[0], 30, 44, 1, 3, 1))

        # train model
        self.model.fit(
            x,
            y_encoded,
            validation_data=(x_val, y_val_encoded),
            epochs=epochs,
            batch_size=batchsize,
            callbacks=[self.early_stop]
        )

        self.__model_trained = True

        return

    def save_model(self, path: str) -> None:
        path_local = Path(path).resolve()

        path_local.mkdir(parents=True, exist_ok=True)

        if self.labels:
            label_path = path_local / "labels.txt"
            with label_path.open("w+") as f:
                for label in self.labels:
                    f.write(f"{label}\n")

        keras_path = path_local / "keras"

        self.model.export(str(keras_path))

        # saved_model_pd = keras_path / "saved_model.pb"

        new_model = tf.saved_model.load(str(keras_path))
        converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
        converter.experimental_enable_resource_variables = True
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]  # pyright: ignore
        tf_model = converter.convert()
        tf_lite_model_path = path_local / "model.tflite"
        with tf_lite_model_path.open("wb") as f:
            f.write(tf_model)

        return

    def load_model(self, path: str):
        path_local = Path(path).resolve()

        if not path_local.exists():
            raise ValueError("Model path does not exist!")
        
        self.model = tf.saved_model.load(str(path_local / "keras"))
        self.__model_trained = True

        label_path = path_local / "labels.txt"

        if label_path.exists():
            with label_path.open("r+") as f:
                self.labels = [label.strip() for label in f.readlines()]

        return
    
    def predict(self, data: dict) -> int | str:
        if not self.__model_trained:
            raise ValueError("The model is not initialized! Either train one or load a model using the load function")

        predict_data = get_predict_data_from_dict(data)

        if predict_data.shape != (30, 44, 3):
            raise ValueError("Data is not of the correct shape!")
        
        pred = np.argmax(self.model.predict(predict_data), axis=1)

        if self.labels:
            return self.labels[pred]

        return pred

    def get_model_stats(self, image_path: Optional[str] = None, labels: Optional[list[str]] = None):
        if not self.__model_trained:
            raise ValueError("Model is not trained yet!")

        Y_pred = self.model.predict(self.x_test)
        y_pred = np.argmax(Y_pred, axis=1)

        print(y_pred)
        print(self.y_test)

        print(classification_report(self.y_test, y_pred))
        fig, ax = plt.subplots(figsize=(13, 12))
        matrix = confusion_matrix(self.y_test, y_pred)
        df = pd.DataFrame(matrix, index=labels, columns=labels)
        
        sns.heatmap(df, ax=ax, annot=True)
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
