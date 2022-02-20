import datetime
import logging
import sys

import numpy as np
import tensorflow as tf
import tf2onnx
import wandb
from data.convnet_dataset import ConvnetDataloader, ConvnetDataset
from data.skeletons_dataset import SkeletonDataset
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from wandb.keras import WandbCallback


def do_train(
    cfg,
    model,
    data,
):

    epochs = cfg.SOLVER.MAX_EPOCHS
    batch_size = cfg.SOLVER.BATCH_SIZE

    logger = logging.getLogger("model.train")
    logger.info("Start training")

    timestamp = datetime.datetime.now().strftime("%d%m%Y%H%M")
    model_name = f"{cfg.MODEL.NAME}_{timestamp}"
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

    callbacks = [
        tensorboard,
        checkpoint,
        WandbCallback(),
        earlyStopping,
        keras.callbacks.ReduceLROnPlateau(),
    ]
    if cfg.MODEL.ARCH == "single":
        X_train_seq, y_train_seq, X_test_seq, y_test_seq = data
        print(y_train_seq.shape)
        history = model.fit(
            X_train_seq,
            y_train_seq,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test_seq, y_test_seq),
            callbacks=callbacks,
        )
    elif cfg.MODEL.ARCH == "double":
        
        (
            X_train_seq_per1,
            X_train_seq_per2,
            y_train_seq,
            X_test_seq_per1,
            X_test_seq_per2,
            y_test_seq,
        ) = data
        print(y_train_seq.shape)
        history = model.fit(
            [X_train_seq_per1, X_train_seq_per2],
            y_train_seq,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=([X_test_seq_per1, X_test_seq_per2], y_test_seq),
            callbacks=callbacks,
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
        ) = data
        history = model.fit(
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
    elif cfg.MODEL.ARCH == "convnet":
        X_train, X_test, y_train, y_test = data

        train_dataset = ConvnetDataset(
            x_data=X_train,
            y_data=y_train,
            linspace_size=wandb.config.number_of_frames,
        )
        val_dataset = ConvnetDataset(
            x_data=X_test,
            y_data=y_test,
            linspace_size=wandb.config.number_of_frames,
        )
        train_dataloader = ConvnetDataloader(
            dataset=train_dataset, batch_size=wandb.config.batch_size, shuffle=True
        )

        val_dataloader = ConvnetDataloader(
            dataset=val_dataset, batch_size=1, shuffle=False
        )

        history = model.fit(
            train_dataloader,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=val_dataloader,
            callbacks=callbacks,
        )
    elif cfg.MODEL.ARCH == "CNN3D":
        train_instances = (
            cfg.DATASETS.TRAIN_CAMERAS
            if cfg.DATASETS.SPLIT_TYPE == "cv"
            else cfg.DATASETS.TRAIN_SUBJECTS
        )
        train_dataset = SkeletonDataset(
            linspace_size=wandb.config.number_of_frames,
            imgs_root_dir=cfg.DATASETS.SKELETON_IMGS,
            train_instances=train_instances,
            data_split=cfg.DATASETS.SPLIT_TYPE,
            train = True,
        )
        val_dataset = SkeletonDataset(
            linspace_size=wandb.config.number_of_frames,
            imgs_root_dir=cfg.DATASETS.SKELETON_IMGS,
            train_instances=train_instances,
            data_split=cfg.DATASETS.SPLIT_TYPE,
            train = False,
        )
        train_dataloader = ConvnetDataloader(
            dataset=train_dataset, batch_size=wandb.config.batch_size, shuffle=True
        )

        val_dataloader = ConvnetDataloader(
            dataset=val_dataset, batch_size=1, shuffle=False
        )
        history = model.fit(
            train_dataloader,
            batch_size=batch_size,
            epochs=epochs,
            # validation_data=val_dataloader,
            callbacks=callbacks,
        )

    # model = keras.models.load_model(filepath)
    # trained_model_artifact = wandb.Artifact(
    #     model_name,
    #     type='model')
    # trained_model_artifact.add_dir('models/')
    # run.log_artifact(trained_model_artifact)
    logger.info("Finished Training")
    logger.info("Saving model ...")
    # model.save(filepath)
    model_proto, _ = tf2onnx.convert.from_keras(
        model, opset=13, output_path=model_name + ".onnx"
    )
    output_names = [n.name for n in model_proto.graph.output]
    print(output_names)

    return model
