import sys

import tensorflow as tf
import wandb
from tensorflow.keras.layers import (
    Flatten,
    MaxPooling3D,
    Dense,
    Dropout,
    Input,
    Conv3D,
    Conv2D,
)
from tensorflow.keras.models import Sequential



class CNN3D:
    """https://www.dbs.ifi.lmu.de/~yu_k/icml2010_3dcnn.pdf"""

    def build_model(input_shape, n_outputs, cfg):
        model = Sequential(
            [
                Input(shape=input_shape),
                Conv3D(
                    46, kernel_size=(7, 7, 3), padding = 'valid', activation="relu"
                ),
                MaxPooling3D(pool_size=(2, 2, 1)),
                Conv3D(
                    46, kernel_size=(7, 6, 3), padding = 'valid', activation="relu"
                ),
                MaxPooling3D(pool_size=(3, 3, 1)),
                Conv3D(
                    78, kernel_size=(3, 3, 1), padding = 'same', activation="relu"
                ),
                Flatten(),
                Dense(128, activation="relu"),
                Dropout(0.5),
                Dense(n_outputs, activation="softmax"),
            ]  
        )

        optimizer = getattr(tf.keras.optimizers, cfg.SOLVER.OPTIMIZER_NAME)
        optimizer = optimizer(lr=wandb.config.learning_rate, decay=1e-6)
        # Compile model
        model.compile(
            loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
        )

        return model
