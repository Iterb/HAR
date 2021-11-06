

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input, Concatenate
import sys 
sys.path.append('.')
class TripleLSTM():
    def build_model(n_timesteps, n_features, n_outputs, n_distance_features, cfg):

        input1 = Input(shape = (n_timesteps,n_features), name = 'input_1')
        lstm1 = LSTM(128, return_sequences=True, name = 'lstm1') (input1)
        dropout1 = Dropout(cfg.MODEL.DROPOUT_RATE) (lstm1)
        lstm12 = LSTM(128, name = 'lstm12',  return_sequences=True) (dropout1)
        dropout12 = Dropout(cfg.MODEL.DROPOUT_RATE) (lstm12)
        lstm13 = LSTM(128, name = 'lstm13') (dropout12)
        dropout13 = Dropout(cfg.MODEL.DROPOUT_RATE) (lstm13)


        input2 = Input(shape = (n_timesteps,n_features), name = 'input_2')
        lstm2 = LSTM(128, return_sequences=True, name = 'lstm2') (input2)
        dropout2 = Dropout(cfg.MODEL.DROPOUT_RATE) (lstm2)
        lstm22 = LSTM(128, return_sequences=True, name = 'lstm22') (dropout2)
        dropout22 = Dropout(cfg.MODEL.DROPOUT_RATE) (lstm22)
        lstm23 = LSTM(128, name = 'lstm23') (dropout22)
        dropout23 = Dropout(cfg.MODEL.DROPOUT_RATE) (lstm23)

        input3 = Input(shape = (n_timesteps, n_distance_features), name = 'input_3')
        lstm3 = LSTM(128, return_sequences=True, name = 'lstm3') (input3)
        dropout3 = Dropout(cfg.MODEL.DROPOUT_RATE) (lstm3)
        lstm32 = LSTM(128, return_sequences=True, name = 'lstm32') (dropout3)
        dropout32 = Dropout(cfg.MODEL.DROPOUT_RATE) (lstm32)
        lstm33 = LSTM(128, name = 'lstm33') (dropout32)
        dropout33 = Dropout(cfg.MODEL.DROPOUT_RATE) (lstm33)


        concat = Concatenate([dropout13, dropout23, dropout33])
        dense = Dense(100, activation= 'relu') (concat)
        output = Dense(n_outputs, activation='softmax') (dense)

        optimizer = getattr(tf.keras.optimizers, cfg.SOLVER.OPTIMIZER_NAME )
        optimizer = optimizer(lr=cfg.SOLVER.BASE_LR, decay=1e-6)

        model = Model(inputs = [input1, input2, input3], outputs = output)
        # Compile model
        model.compile(
            loss='categorical_crossentropy',
            optimizer= optimizer,
            metrics=['accuracy']
        )

 
        return model
