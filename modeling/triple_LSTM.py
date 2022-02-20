import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input, concatenate
import sys
import wandb
sys.path.append(".")


class TripleLSTM:
    def build_model(n_timesteps, n_features, n_outputs, n_distance_features, cfg):
        lstm_size = wandb.config.LSTM_size
        dropout_rate = wandb.config.dropout
        input1 = Input(shape=(n_timesteps, n_features), name="input_1")
        lstm1 = LSTM(lstm_size, return_sequences=True, name="lstm1")(input1)
        dropout1 = Dropout(dropout_rate)(lstm1)
        lstm12 = LSTM(lstm_size, name="lstm12", return_sequences=False)(dropout1)
        dropout12 = Dropout(dropout_rate)(lstm12)
        # lstm13 = LSTM(lstm_size, name="lstm13")(dropout12)
        # dropout13 = Dropout(dropout_rate)(lstm13)

        input2 = Input(shape=(n_timesteps, n_features), name="input_2")
        lstm2 = LSTM(lstm_size, return_sequences=True, name="lstm2")(input2)
        dropout2 = Dropout(dropout_rate)(lstm2)
        lstm22 = LSTM(lstm_size, return_sequences=False, name="lstm22")(dropout2)
        dropout22 = Dropout(dropout_rate)(lstm22)
        # lstm23 = LSTM(lstm_size, name="lstm23")(dropout22)
        # dropout23 = Dropout(dropout_rate)(lstm23)

        input3 = Input(shape=(n_timesteps, n_distance_features), name="input_3")
        lstm3 = LSTM(lstm_size, return_sequences=True, name="lstm3")(input3)
        dropout3 = Dropout(dropout_rate)(lstm3)
        lstm32 = LSTM(lstm_size, return_sequences=False, name="lstm32")(dropout3)
        dropout32 = Dropout(dropout_rate)(lstm32)
        # lstm33 = LSTM(lstm_size, name="lstm33")(dropout32)
        # dropout33 = Dropout(dropout_rate)(lstm33)
        concat = concatenate([dropout12, dropout22, dropout32])
        dense = Dense(100, activation="relu")(concat)
        output = Dense(n_outputs, activation="softmax")(dense)

        optimizer = getattr(tf.keras.optimizers, cfg.SOLVER.OPTIMIZER_NAME)
        optimizer = optimizer(lr=cfg.SOLVER.BASE_LR, decay=1e-6)

        model = Model(inputs=[input1, input2, input3], outputs=output)
        # Compile model
        model.compile(
            loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
        )

        return model
