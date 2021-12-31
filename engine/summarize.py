import json
import logging
from typing import Dict

import numpy as np
import seaborn as sns
import wandb
from sklearn.metrics import classification_report, confusion_matrix

def create_wandb_table(classification_report: Dict[str, Dict[str, float]])->wandb.Table:
    columns = ["class"] + list(classification_report['punch'].keys())
    table = wandb.Table(columns=columns)
    for _class, values in classification_report.items():
        metric_values = list(values.keys())
        table.add_data(_class, metric_values[0],metric_values[1],metric_values[2],metric_values[3])

    return table
def summarize(
    cfg,
    model,
    data,
):

    logger = logging.getLogger("model.summarize")
    logger.info("Getting results")
    class_dict = {
        "punch": 0,
        "kicking": 1,
        "pushing": 2,
        "pat on back": 3,
        "point finger": 4,
        "hugging": 5,
        "giving an object": 6,
        "touch pocket": 7,
        "shaking hands": 8,
        "walking towards": 9,
        "walking apart": 10,
    }
    if cfg.MODEL.ARCH == "single":
        X_train_seq, y_train_seq, X_test_seq, y_test_seq = data
        Y_pred = model.predict(X_test_seq, verbose=1)
    elif cfg.MODEL.ARCH == "double":
        (
            X_train_seq_per1,
            X_train_seq_per2,
            y_train_seq,
            X_test_seq_per1,
            X_test_seq_per2,
            y_test_seq,
        ) = data
        Y_pred = model.predict([X_test_seq_per1, X_test_seq_per2], verbose=1)
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
        Y_pred = model.predict(
            [X_test_seq_per1, X_test_seq_per2, X_test_dist_seq], verbose=1
        )
    y_pred = np.argmax(Y_pred, axis=1)
    _y = np.argmax(y_test_seq, axis=1)
    cf_matrix = confusion_matrix(_y, y_pred)
    report = classification_report(
        _y, y_pred, target_names=list(class_dict.keys()), output_dict=True
    )
    sns.set(rc={"figure.figsize": (16, 12)})
    heatmap = sns.heatmap(
        cf_matrix,
        annot=True,
        fmt="g",
        cmap="Blues",
        xticklabels=list(class_dict.keys()),
        yticklabels=list(class_dict.keys()),
    )
    images = wandb.Image(heatmap, caption="Top: Output, Bottom: Input")
    wandb.log({"Confusion_matrix": images})
    print(report)
    wandb.log({"Classification_report_table":create_wandb_table(report)})
    wandb.log({"Classification_report":report})

    logger.info("Got results")
