import tensorflow as tf
import wandb
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
)
from tensorflow.keras.models import Sequential


class SingleLSTM:
    def __init__(self):
        self.model = Sequential()

    def build_model(self, n_timesteps, n_features, n_outputs, cfg):

        if wandb.config.num_of_lstm_layers == 1:
            self._LSTM_n_Dropout(
                wandb.config.LSTM_size,
                return_sequences=False,
                dropout=wandb.config.dropout,
            )
        if wandb.config.num_of_lstm_layers == 2:
            self._LSTM_n_Dropout(
                wandb.config.LSTM_size,
                return_sequences=True,
                dropout=wandb.config.dropout,
            )
            self._LSTM_n_Dropout(
                wandb.config.LSTM_size,
                return_sequences=False,
                dropout=wandb.config.dropout,
            )
        if wandb.config.num_of_lstm_layers == 3:
            self._LSTM_n_Dropout(
                wandb.config.LSTM_size,
                return_sequences=True,
                dropout=wandb.config.dropout,
            )
            self._LSTM_n_Dropout(
                wandb.config.LSTM_size,
                return_sequences=True,
                dropout=wandb.config.dropout,
            )
            self._LSTM_n_Dropout(
                wandb.config.LSTM_size,
                return_sequences=False,
                dropout=wandb.config.dropout,
            )

        self.model.add(Dense(100, activation="relu"))
        self.model.add(Dropout(wandb.config.dropout))

        self.model.add(Dense(n_outputs, activation="softmax"))

        optimizer = getattr(tf.keras.optimizers, cfg.SOLVER.OPTIMIZER_NAME)
        optimizer = optimizer(lr=wandb.config.learning_rate, decay=1e-6)
        # Compile self.model
        self.model.compile(
            loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
        )

        return self.model

    def _LSTM_n_Dropout(self, LSTM_size, return_sequences, dropout):
        self.model.add(LSTM(LSTM_size, return_sequences=return_sequences))
        self.model.add(Dropout(dropout))
