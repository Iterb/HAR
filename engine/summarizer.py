import logging
from typing import Dict

import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import wandb


def create_wandb_table(
    classification_report: Dict[str, Dict[str, float]]
) -> wandb.Table:
    columns = ["class"] + list(classification_report["punch"].keys())
    table = wandb.Table(columns=columns)
    for _class, values in classification_report.items():
        if isinstance(values, float):
            continue
        metric_values = list(values.values())
        table.add_data(_class, *metric_values)

    return table


class Summarizer:
    def __init__(self, model, dataset, cfg) -> None:
        self.cfg = cfg
        self.model = model
        self.dataset = dataset

        logger = logging.getLogger("model.summarize")
        logger.info("Getting results")
        self.class_dict = {
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
            X_train_seq, y_train_seq, X_test_seq, y_test_seq = dataset.process_data()
            self.results = model.predict(X_test_seq, verbose=1)
        elif cfg.MODEL.ARCH == "double":
            (
                X_train_seq_per1,
                X_train_seq_per2,
                y_train_seq,
                X_test_seq_per1,
                X_test_seq_per2,
                y_test_seq,
            ) = dataset.process_data()
            self.results = model.predict([X_test_seq_per1, X_test_seq_per2], verbose=1)
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
            self.results = model.predict(
                [X_test_seq_per1, X_test_seq_per2, X_test_dist_seq], verbose=1
            )
        self.y_pred = np.argmax(self.results, axis=1)
        self._y = np.argmax(y_test_seq, axis=1)

    def classification_report(self):
        report = classification_report(
            self._y,
            self.y_pred,
            target_names=list(self.class_dict.keys()),
            output_dict=True,
        )
        wandb.log({"Classification_report_table": create_wandb_table(report)})
        wandb.log({"Classification_report": report})

        return report

    def create_cm(self):
        self.cf_matrix = confusion_matrix(self._y, self.y_pred)
        return self.cf_matrix

    def vizualize(self):
        sns.set(rc={"figure.figsize": (16, 12)})
        heatmap = sns.heatmap(
            self.cf_matrix,
            annot=True,
            fmt="g",
            cmap="Blues",
            xticklabels=list(self.class_dict.keys()),
            yticklabels=list(self.class_dict.keys()),
        )
        images = wandb.Image(heatmap, caption="Top: Output, Bottom: Input")
        wandb.log({"Confusion_matrix": images})
