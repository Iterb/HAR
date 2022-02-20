import sys

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, concatenate
from tensorflow.keras.models import Model

import wandb

sys.path.append(".")


class DoubleLSTM:

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = self._build_model()

    def _build_model(self):
        if self.cfg.DATASETS.FEATURES_TYPE == 1:
            f_size = 41
        elif self.cfg.DATASETS.FEATURES_TYPE == 2:
            f_size = 450
        elif self.cfg.DATASETS.FEATURES_TYPE == 3:
            f_size = 30
        input1 = Input(
            shape=(self.cfg.SEQUENCE.WINDOW_SIZE, f_size),
            name="input_1",
        )
        lstm1 = LSTM(wandb.config.LSTM_size, return_sequences=True, name="lstm1")(
            input1
        )
        dropout1 = Dropout(wandb.config.dropout)(lstm1)
        lstm12 = LSTM(wandb.config.LSTM_size, name="lstm12", return_sequences=False)(
            dropout1
        )
        dropout12 = Dropout(
            wandb.config.dropout,
        )(lstm12)

        input2 = Input(
            shape=(self.cfg.SEQUENCE.WINDOW_SIZE, f_size),
            name="input_2",
        )
        lstm2 = LSTM(wandb.config.LSTM_size, return_sequences=True, name="lstm2")(
            input2
        )
        dropout2 = Dropout(
            wandb.config.dropout,
        )(lstm2)
        lstm22 = LSTM(wandb.config.LSTM_size, return_sequences=False, name="lstm22")(
            dropout2
        )
        dropout22 = Dropout(
            wandb.config.dropout,
        )(lstm22)

        concat = concatenate([dropout12, dropout22])

        dense = Dense(100, activation="relu")(concat)
        output = Dense(11, activation="softmax")(dense)
        return Model(inputs=[input1, input2], outputs=output)

    def compile_model(self):
        optimizer = getattr(tf.keras.optimizers, self.cfg.SOLVER.OPTIMIZER_NAME)
        optimizer = optimizer(lr=wandb.config.learning_rate, decay=1e-6)
        # Compile model
        self.model.compile(
            loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
        )

        return self.model
