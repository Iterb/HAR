import logging
from pickle import load
import datetime
from collections import deque

import wandb
import tensorflow as tf
import numpy as np
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
import onnx
import onnxruntime as rt
import pandas as pd

FEATURES_TYPE_1 = 1
FEATURES_TYPE_2 = 2
FEATURES_TYPE_3 = 3
SINGLE_LSTM = "single"
DOUBLE_LSTM = "double"
TRIPLE_LSTM = "triple"


def do_inference(
    cfg,
    data,
):

    m = rt.InferenceSession(cfg.INFER.MODEL_PATH)
    features = preprocess(cfg, data)
    logger = logging.getLogger("model.infer")
    logger.info("Start inferencing")
    input_name = m.get_inputs()[0].name
    label_name = m.get_outputs()[0].name
    features = features[0].astype(np.float32).reshape(-1, cfg.INFER.WINDOW_SIZE, 60)
    onnx_pred = m.run([label_name], {input_name: features})
    class_dict = {
        0: "punch",
        1: "kicking",
        2: "pushing",
        3: "pat on back",
        4: "point finger",
        5: "hugging",
        6: "giving an object",
        7: "touch pocket",
        8: "shaking hands",
        9: "walking towards",
        10: "walking apart",
    }
    preds = np.argmax((onnx_pred), axis=2)
    for p in preds[0]:
        print(class_dict[p])
    logger.info("Finished Testing")
    return onnx_pred


def preprocess(cfg, data):
    scaler = load(
        open(f"sklearn/{cfg.MODEL.ARCH}_F{cfg.DATASETS.FEATURES_TYPE}_scaler.pkl", "rb")
    )
    imputer = load(
        open(
            f"sklearn/{cfg.MODEL.ARCH}_F{cfg.DATASETS.FEATURES_TYPE}_imputer.pkl", "rb"
        )
    )
    seq_type = "linspace"
    if cfg.INFER.ARCH == SINGLE_LSTM:
        features = data
        features = pd.DataFrame(
            imputer.transform(features), columns=features.columns, index=features.index
        )
        features = pd.DataFrame(
            scaler.transform(features), columns=features.columns, index=features.index
        )
        temporal_features = create_sequence_for_infer(
            cfg.INFER.WINDOW_SIZE, features, seq_type
        )
        return [temporal_features]
    if cfg.INFER.ARCH == DOUBLE_LSTM:
        features_per1, features_per2 = data
        features_per1 = pd.DataFrame(
            imputer.transform(features_per1),
            columns=features_per1.columns,
            index=features_per1.index,
        )
        features_per1 = pd.DataFrame(
            scaler.transform(features_per1),
            columns=features_per1.columns,
            index=features_per1.index,
        )
        features_per2 = pd.DataFrame(
            imputer.transform(features_per2),
            columns=features_per2.columns,
            index=features_per2.index,
        )
        features_per2 = pd.DataFrame(
            scaler.transform(features_per2),
            columns=features_per2.columns,
            index=features_per2.index,
        )
        temporal_features_per1 = create_sequence_for_infer(
            cfg.INFER.WINDOW_SIZE, features_per1, seq_type
        )
        temporal_features_per2 = create_sequence_for_infer(
            cfg.INFER.WINDOW_SIZE, features_per2, seq_type
        )
        return [temporal_features_per1, temporal_features_per2]
    if cfg.INFER.ARCH == TRIPLE_LSTM:
        scaler_dist = load(
            open(
                f"sklearn/{cfg.MODEL.ARCH}_F{cfg.DATASETS.FEATURES_TYPE}_scaler_dist.pkl",
                "rb",
            )
        )
        imputer_dist = load(
            open(
                f"sklearn/{cfg.MODEL.ARCH}_F{cfg.DATASETS.FEATURES_TYPE}_imputer_dist.pkl",
                "rb",
            )
        )
        features_per1, features_per2, dist = data
        features_per1 = pd.DataFrame(
            imputer.transform(features_per1),
            columns=features_per1.columns,
            index=features_per1.index,
        )
        features_per1 = pd.DataFrame(
            scaler.transform(features_per1),
            columns=features_per1.columns,
            index=features_per1.index,
        )
        features_per2 = pd.DataFrame(
            imputer.transform(features_per2),
            columns=features_per2.columns,
            index=features_per2.index,
        )
        features_per2 = pd.DataFrame(
            scaler.transform(features_per2),
            columns=features_per2.columns,
            index=features_per2.index,
        )
        dist = pd.DataFrame(
            imputer_dist.transform(dist), columns=dist.columns, index=dist.index
        )
        dist = pd.DataFrame(
            scaler_dist.transform(dist), columns=dist.columns, index=dist.index
        )
        temporal_features_per1 = create_sequence_for_infer(
            cfg.INFER.WINDOW_SIZE, features_per1, seq_type
        )
        temporal_features_per2 = create_sequence_for_infer(
            cfg.INFER.WINDOW_SIZE, features_per2, seq_type
        )
        temporal_features_dist = create_sequence_for_infer(
            cfg.INFER.WINDOW_SIZE, dist, seq_type
        )
        return [temporal_features_per1, temporal_features_per2, temporal_features_dist]


def create_sequence_for_infer(window_size, df, seq_type):

    if seq_type == "window":
        sequential_data = []
        prev_poses_data = deque(maxlen=window_size)
        # print (df)
        for i in df.values:
            prev_poses_data.append(i)
            if len(prev_poses_data) == window_size:
                sequential_data.append(prev_poses_data)
        return np.array(sequential_data)
    elif seq_type == "linspace":
        idx = np.round(np.linspace(0, len(df) - 1, window_size)).astype(int)
        df = df.iloc[idx, :].reset_index(drop=True)
        sequential_data = [np.array(df)]
        return np.array(sequential_data)
