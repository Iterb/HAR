import logging
import numpy as np
import datetime
import wandb
from wandb.keras import WandbCallback
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
def do_train(
        cfg,
        model,
        data,
        run,
):

    epochs = cfg.SOLVER.MAX_EPOCHS
    batch_size = cfg.SOLVER.BATCH_SIZE
    
    logger = logging.getLogger("model.train")
    logger.info("Start training")

    timestamp = datetime.datetime.now().strftime("%d%m%Y%H%M")
    tensorboard = TensorBoard(log_dir=f"logs/{cfg.MODEL.NAME}_{timestamp}")

    filepath = f"models/{cfg.MODEL.NAME}_{timestamp}.h5" 

    checkpoint = ModelCheckpoint(
        filepath=filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    earlyStopping =  EarlyStopping(
        monitor='val_accuracy',
        min_delta=0,
        patience=5,
        verbose=0,
        mode='auto',
        baseline=None,
        restore_best_weights=True
)
    X_train_seq, y_train_seq, X_test_seq, y_test_seq = data

    history = model.fit(
        X_train_seq, y_train_seq,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test_seq, y_test_seq),
        callbacks=[tensorboard, checkpoint, WandbCallback(), earlyStopping]
    )

    trained_model_artifact = wandb.Artifact(
        cfg.MODEL.NAME, 
        type='model')
    trained_model_artifact.add_dir(filepath)
    run.log_artifact(trained_model_artifact)
    logger.info('Finished Training')
    logger.info('Saving model ...')


    return model