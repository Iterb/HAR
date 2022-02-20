import datetime
import logging

import tf2onnx
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from wandb.keras import WandbCallback


class Trainer:
    def __init__(self, model, dataset, config) -> None:
        self.model = model
        self.dataset = dataset
        self.config = config

    def fit(self):
        epochs = self.config.SOLVER.MAX_EPOCHS
        batch_size = self.config.SOLVER.BATCH_SIZE

        logger = logging.getLogger("model.train")
        logger.info("Start training")

        timestamp = datetime.datetime.now().strftime("%d%m%Y%H%M")
        self.model_name = f"{self.config.MODEL.NAME}_{timestamp}"
        tensorboard = TensorBoard(log_dir=f"logs/{self.config.MODEL.NAME}_{timestamp}")

        filepath = f"models/{self.model_name}"

        checkpoint = ModelCheckpoint(
            filepath=filepath,
            save_weights_only=True,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        )

        earlyStopping = EarlyStopping(
            monitor="val_accuracy",
            min_delta=0,
            patience=7,
            verbose=0,
            mode="auto",
            baseline=None,
            restore_best_weights=True,
        )

        callbacks = [
            tensorboard,
            checkpoint,
            WandbCallback(),
            earlyStopping,
            keras.callbacks.ReduceLROnPlateau(),
        ]
        if self.config.MODEL.ARCH == "single":
            (
                X_train_seq,
                y_train_seq,
                X_test_seq,
                y_test_seq,
            ) = self.dataset.process_data()

            history = self.model.fit(
                X_train_seq,
                y_train_seq,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_test_seq, y_test_seq),
                callbacks=callbacks,
            )
        elif self.config.MODEL.ARCH == "double":

            (
                X_train_seq_per1,
                X_train_seq_per2,
                y_train_seq,
                X_test_seq_per1,
                X_test_seq_per2,
                y_test_seq,
            ) = self.dataset.process_data()
            print(y_train_seq.shape)
            history = self.model.fit(
                [X_train_seq_per1, X_train_seq_per2],
                y_train_seq,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=([X_test_seq_per1, X_test_seq_per2], y_test_seq),
                callbacks=callbacks,
            )
        elif self.config.MODEL.ARCH == "triple":
            (
                X_train_seq_per1,
                X_train_seq_per2,
                X_train_dist_seq,
                X_test_seq_per1,
                X_test_seq_per2,
                X_test_dist_seq,
                y_train_seq,
                y_test_seq,
            ) = self.dataset.process_data()
            history = self.model.fit(
                [X_train_seq_per1, X_train_seq_per2, X_train_dist_seq],
                y_train_seq,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(
                    [X_test_seq_per1, X_test_seq_per2, X_test_dist_seq],
                    y_test_seq,
                ),
                callbacks=callbacks,
            )

        logger.info("Finished Training")
        return self.model

    def onnx_export(self):

        model_proto, _ = tf2onnx.convert.from_keras(
            self.model, opset=13, output_path=f"{self.model_name}.onnx"
        )

        return model_proto
