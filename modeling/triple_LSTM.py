import sys

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, concatenate
from tensorflow.keras.models import Model

import wandb

sys.path.append(".")


class TripleLSTM:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = self._build_model()

    def _build_model(self):
        lstm_size = wandb.config.LSTM_size
        dropout_rate = wandb.config.dropout
        input1 = Input(shape=(self.cfg.SEQUENCE.WINDOW_SIZE, 26), name="input_1")
        lstm1 = LSTM(lstm_size, return_sequences=True, name="lstm1")(input1)
        dropout1 = Dropout(dropout_rate)(lstm1)
        lstm12 = LSTM(lstm_size, name="lstm12", return_sequences=False)(dropout1)
        dropout12 = Dropout(dropout_rate)(lstm12)

        input2 = Input(shape=(self.cfg.SEQUENCE.WINDOW_SIZE, 26), name="input_2")
        lstm2 = LSTM(lstm_size, return_sequences=True, name="lstm2")(input2)
        dropout2 = Dropout(dropout_rate)(lstm2)
        lstm22 = LSTM(lstm_size, return_sequences=False, name="lstm22")(dropout2)
        dropout22 = Dropout(dropout_rate)(lstm22)

        input3 = Input(shape=(self.cfg.SEQUENCE.WINDOW_SIZE, 25), name="input_3")
        lstm3 = LSTM(lstm_size, return_sequences=True, name="lstm3")(input3)
        dropout3 = Dropout(dropout_rate)(lstm3)
        lstm32 = LSTM(lstm_size, return_sequences=False, name="lstm32")(dropout3)
        dropout32 = Dropout(dropout_rate)(lstm32)

        concat = concatenate([dropout12, dropout22, dropout32])
        dense = Dense(100, activation="relu")(concat)
        output = Dense(11, activation="softmax")(dense)

        return Model(inputs=[input1, input2, input3], outputs=output)

    def compile_model(self):

        optimizer = getattr(tf.keras.optimizers, self.cfg.SOLVER.OPTIMIZER_NAME)
        optimizer = optimizer(lr=self.cfg.SOLVER.BASE_LR, decay=1e-6)
        # Compile model
        self.model.compile(
            loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
        )

        return self.model
