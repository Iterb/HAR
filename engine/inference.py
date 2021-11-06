import logging
from pickle import load
import datetime
from collections import deque

import wandb
import tensorflow as tf
import numpy as np
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
import pandas as pd
FEATURES_TYPE_1 = 1
FEATURES_TYPE_2 = 2
FEATURES_TYPE_3 = 3
SINGLE_LSTM = 'single'
DOUBLE_LSTM = 'double'
TRIPLE_LSTM = 'triple'

def do_test(
        cfg,
        model,
        data,
):
    logger = logging.getLogger("model.test")
    logger.info("Start testing")
    X_train_seq, y_train_seq, X_test_seq, y_test_seq = data

    score = model.evaluate(X_test_seq, y_test_seq, verbose=1)

    wandb.log({'Best_val_acc': score[1]})
    logger.info('Finished Testing')



    return score

def preprocess(cfg, data):
    scaler = load(open(f'sklearn/{cfg.MODEL.ARCH}_F{cfg.DATASETS.FEATURES_TYPE}_scaler.pkl', 'rb'))
    imputer = load(open(f'sklearn/{cfg.MODEL.ARCH}_F{cfg.DATASETS.FEATURES_TYPE}_imputer.pkl', 'rb'))

    if cfg.INFER.ARCH == SINGLE_LSTM:
        features = data
        features =  pd.DataFrame(imputer.transform(features), columns=features.columns, index=features.index)
        features = pd.DataFrame(scaler.transform(features), columns=features.columns, index=features.index)
        temporal_features = create_sequence_for_infer(cfg.INFER.WINDOW_SIZE, features)
        return [temporal_features]
    if cfg.INFER.ARCH == DOUBLE_LSTM:
        features_per1, features_per2 = data
        features_per1 =  pd.DataFrame(imputer.transform(features_per1), columns=features_per1.columns, index=features_per1.index)
        features_per1 = pd.DataFrame(scaler.transform(features_per1), columns=features_per1.columns, index=features_per1.index)
        features_per2 =  pd.DataFrame(imputer.transform(features_per2), columns=features_per2.columns, index=features_per2.index)
        features_per2 = pd.DataFrame(scaler.transform(features_per2), columns=features_per2.columns, index=features_per2.index)
        temporal_features_per1 = create_sequence_for_infer(cfg.INFER.WINDOW_SIZE, features_per1)
        temporal_features_per2 = create_sequence_for_infer(cfg.INFER.WINDOW_SIZE, features_per2)
        return [temporal_features_per1, temporal_features_per2]
    if cfg.INFER.ARCH == TRIPLE_LSTM:
        scaler_dist = load(open(f'sklearn/{cfg.MODEL.ARCH}_F{cfg.DATASETS.FEATURES_TYPE}_scaler_dist.pkl', 'rb'))
        imputer_dist = load(open(f'sklearn/{cfg.MODEL.ARCH}_F{cfg.DATASETS.FEATURES_TYPE}_imputer_dist.pkl', 'rb'))
        features_per1, features_per2, dist = data
        features_per1 =  pd.DataFrame(imputer.transform(features_per1), columns=features_per1.columns, index=features_per1.index)
        features_per1 = pd.DataFrame(scaler.transform(features_per1), columns=features_per1.columns, index=features_per1.index)
        features_per2 =  pd.DataFrame(imputer.transform(features_per2), columns=features_per2.columns, index=features_per2.index)
        features_per2 = pd.DataFrame(scaler.transform(features_per2), columns=features_per2.columns, index=features_per2.index)
        dist = pd.DataFrame(imputer_dist.transform(dist), columns=dist.columns, index=dist.index)
        dist = pd.DataFrame(scaler_dist.transform(dist), columns=dist.columns, index=dist.index)
        temporal_features_per1 = create_sequence_for_infer(cfg.INFER.WINDOW_SIZE, features_per1)
        temporal_features_per2 = create_sequence_for_infer(cfg.INFER.WINDOW_SIZE, features_per2)
        temporal_features_dist = create_sequence_for_infer(cfg.INFER.WINDOW_SIZE, dist)
        return [temporal_features_per1, temporal_features_per2, temporal_features_dist]
        
def create_sequence_for_infer(window_size, df):

    sequential_data = []
    prev_poses_data = deque(maxlen = window_size)
        #print (df)
    for i in df.values:
        prev_poses_data.append(i)
        if len(prev_poses_data) == window_size:
            sequential_data.append(np.array(prev_poses_data))

    return np.array(sequential_data)