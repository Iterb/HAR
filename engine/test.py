import logging
import numpy as np
import datetime
import wandb
import tensorflow as tf


def do_test(
    cfg,
    model,
    data,
):

    logger = logging.getLogger("model.test")
    logger.info("Start testing")
    if cfg.MODEL.ARCH == "single":
        X_train_seq, y_train_seq, X_test_seq, y_test_seq = data
        score = model.evaluate(X_test_seq, y_test_seq, verbose=1)
    elif cfg.MODEL.ARCH == "double":
        (
            X_train_seq_per1,
            X_train_seq_per2,
            y_train_seq,
            X_test_seq_per1,
            X_test_seq_per2,
            y_test_seq,
        ) = data
        score = model.evaluate(
            [X_test_seq_per1, X_test_seq_per2], y_test_seq, verbose=1
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
        score = model.evaluate(
            [X_test_seq_per1, X_test_seq_per2, X_test_dist_seq], y_test_seq, verbose=1
        )
    wandb.log({"Best_val_acc": score[1]})
    logger.info("Finished Testing")

    return score
