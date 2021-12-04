import sys

import tensorflow as tf
import wandb
from tensorflow.keras.layers import (
    LSTM,
    Average,
    BatchNormalization,
    Dense,
    Dropout,
    Input,
    concatenate,
)
from tensorflow.keras.models import Model, Sequential

sys.path.append(".")
from config import cfg
from data.dataset import Dataset


class SingleLSTM:
    def build_model(n_timesteps, n_features, n_outputs, cfg):

        model = Sequential()
        model.add(
            LSTM(128, input_shape=(n_timesteps, n_features), return_sequences=True)
        )
        model.add(Dropout(wandb.config.dropout))
        if wandb.config.num_of_lstm_layers in [2, 3]:
            model.add(LSTM(128, return_sequeSnces=True))
            model.add(Dropout(wandb.config.dropout))
        if wandb.config.num_of_lstm_layers == 3:
            model.add(LSTM(128))
            model.add(Dropout(wandb.config.dropout))

        model.add(Dense(100, activation="relu"))
        model.add(Dropout(wandb.config.dropout))

        model.add(Dense(n_outputs, activation="softmax"))

        optimizer = getattr(tf.keras.optimizers, cfg.SOLVER.OPTIMIZER_NAME)
        optimizer = optimizer(lr=wandb.config.learning_rate, decay=1e-6)
        # Compile model
        model.compile(
            loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
        )

        return model
