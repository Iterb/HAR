import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input, concatenate
import sys
import wandb
sys.path.append(".")


class DoubleLSTM:
    def build_model(n_timesteps, n_features, n_outputs, cfg):

        input1 = Input(shape=(n_timesteps, n_features), name="input_1")
        lstm1 = LSTM(wandb.config.LSTM_size, return_sequences=True, name="lstm1")(input1)
        dropout1 = Dropout(wandb.config.dropout)(lstm1)
        lstm12 = LSTM(wandb.config.LSTM_size, name="lstm12", return_sequences=False)(dropout1)
        dropout12 = Dropout(wandb.config.dropout,)(lstm12)


        input2 = Input(shape=(n_timesteps, n_features), name="input_2")
        lstm2 = LSTM(wandb.config.LSTM_size, return_sequences=True, name="lstm2")(input2)
        dropout2 = Dropout(wandb.config.dropout,)(lstm2)
        lstm22 = LSTM(wandb.config.LSTM_size, return_sequences=False, name="lstm22")(dropout2)
        dropout22 = Dropout(wandb.config.dropout,)(lstm22)


        concat = concatenate([dropout12, dropout22])

        dense = Dense(100, activation="relu")(concat)
        output = Dense(n_outputs, activation="softmax")(dense)

        model = Model(inputs=[input1, input2], outputs=output)

        optimizer = getattr(tf.keras.optimizers, cfg.SOLVER.OPTIMIZER_NAME)
        optimizer = optimizer(lr=wandb.config.learning_rate, decay=1e-6)
        # Compile model
        model.compile(
            loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
        )

        return model
