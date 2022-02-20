import datetime
import logging

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from wandb.keras import WandbCallback


def do_sweep(
    cfg,
    model,
    dataset,
):

    epochs = cfg.SOLVER.MAX_EPOCHS
    batch_size = cfg.SOLVER.BATCH_SIZE

    logger = logging.getLogger("model.train")
    logger.info("Start training")

    timestamp = datetime.datetime.now().strftime("%d%m%Y%H%M")
    model_name = f"{cfg.MODEL.NAME}_{timestamp}.h5"
    tensorboard = TensorBoard(log_dir=f"logs/{cfg.MODEL.NAME}_{timestamp}")

    filepath = f"models/{model_name}"

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

    if cfg.MODEL.ARCH == "single":
        X_train_seq, y_train_seq, X_test_seq, y_test_seq = dataset.process_data()
        history = model.fit(
            X_train_seq,
            y_train_seq,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test_seq, y_test_seq),
            callbacks=[tensorboard, checkpoint, WandbCallback(), earlyStopping],
        )
    elif cfg.MODEL.ARCH == "double":
        (
            X_train_seq_per1,
            X_train_seq_per2,
            y_train_seq,
            X_test_seq_per1,
            X_test_seq_per2,
            y_test_seq,
        ) = dataset.process_data()
        history = model.fit(
            [X_train_seq_per1, X_train_seq_per2],
            y_train_seq,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=([X_test_seq_per1, X_test_seq_per2], y_test_seq),
            callbacks=[tensorboard, checkpoint, WandbCallback(), earlyStopping],
        )
    elif cfg.MODEL.ARCH == "triple":
        (
            X_train_seq_per1,
            X_train_seq_per2,
            X_train_dist_seq,
            X_test_seq_per1,
            X_test_seq_per2,
            X_test_dist_seq,
            y_train_seq,
            y_test_seq,
        ) = dataset.process_data()
        history = model.fit(
            [X_train_seq_per1, X_train_seq_per2, X_train_dist_seq],
            y_train_seq,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(
                [X_test_seq_per1, X_test_seq_per2, X_test_dist_seq],
                y_test_seq,
            ),
            callbacks=[tensorboard, checkpoint, WandbCallback(), earlyStopping],
        )
    # model = keras.models.load_model(filepath)
    # trained_model_artifact = wandb.Artifact(
    #     model_name,
    #     type='model')
    # trained_model_artifact.add_dir('models/')
    # run.log_artifact(trained_model_artifact)
    logger.info("Finished Training")
    logger.info("Saving model ...")
