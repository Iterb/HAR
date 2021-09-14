
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, concatenate, Input, Average
import sys 
sys.path.append('.')
from config import cfg
from data.dataset import Dataset
class SingleLSTM():
    def build_model(n_timesteps, n_features, n_outputs, cfg):
        model = Sequential()
        model.add(LSTM(128, input_shape=(n_timesteps,n_features), return_sequences=True))
        model.add(Dropout(0.2))

        model.add(LSTM(128, return_sequences=True))
        model.add(Dropout(0.2))

        model.add(LSTM(128))
        model.add(Dropout(0.2))

        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(n_outputs, activation='softmax'))

        
        optimizer = getattr(tf.keras.optimizers, cfg.SOLVER.OPTIMIZER_NAME )
        optimizer = optimizer(lr=cfg.SOLVER.BASE_LR, decay=1e-6)
        # Compile model
        model.compile(
            loss='categorical_crossentropy',
            optimizer= optimizer,
            metrics=['accuracy']
        )

 
        return model

    

    def evaluate_model(model, X_test_seq, y_test_seq):
        score = model.evaluate(X_test_seq, y_test_seq, verbose=1)
        return score[1]


if __name__ == '__main__':

        dataset = Dataset(cfg)
        # tensorboard = TensorBoard(log_dir=f"logs/{NAME}")

        # filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
        # checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')) # saves only the best ones
        X_train_seq, y_train_seq, X_test_seq, y_test_seq = dataset.create_test_train_sets()
        model = SingleLSTM.build_model(X_train_seq.shape[1], X_train_seq.shape[2], y_train_seq.shape[1], cfg)
        history = model.fit(
            X_train_seq, y_train_seq,
            batch_size=64,
            epochs=20,
            validation_data=(X_test_seq, y_test_seq),
            #callbacks=[tensorboard],
        )