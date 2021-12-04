# encoding: utf-8

from collections import deque

import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical


def create_sequences(x, y, cfg):

    df = pd.concat([x, y], axis=1)
    # sequential_data = []
    prev_poses_data = deque(maxlen=cfg.SEQUENCE.WINDOW_SIZE)
    current_batch = 0
    X = []
    y = []
    # print (df)
    for i in df.values:
        prev_poses_data.append([n for n in i[:-2]])
        if current_batch == (i[-2]):
            if len(prev_poses_data) == cfg.SEQUENCE.WINDOW_SIZE:
                # sequential_data.append([np.array(prev_poses_data), i[-1]])
                X.append(np.array(prev_poses_data))
                y.append(i[-1])
        else:
            current_batch = i[-2]
            prev_poses_data = deque(maxlen=cfg.SEQUENCE.WINDOW_SIZE)

            prev_poses_data.append([n for n in i[:-2]])

    #   for seq, label in sequential_data:
    #     X.append(seq)
    #     y.append(label-LABEL_OFFSET)
    X = np.array(X)
    y = np.array(y).reshape(X.shape[0], 1)
    return np.array(X), to_categorical(np.array(y))


def create_sequences2(x, y, cfg):

    df = pd.concat([x, y], axis=1)
    prev_poses_data = deque(maxlen=cfg.SEQUENCE.INDOW_SIZE)
    current_batch = 0
    X = []
    y = []
    batch = []

    for i in df.values:
        prev_poses_data.append([n for n in i[:-2]])
        if current_batch == (i[-2]):
            if len(prev_poses_data) == cfg.SEQUENCE.WINDOW_SIZE:
                # sequential_data.append([np.array(prev_poses_data), i[-1]])
                X.append(np.array(prev_poses_data))
                y.append(i[-1])
        else:
            batch.append([X, y])
            X = []
            y = []
            current_batch = i[-2]
            prev_poses_data = deque(maxlen=cfg.SEQUENCE.WINDOW_SIZE)

            prev_poses_data.append([n for n in i[:-2]])

    return batch


def create_spaced_sequences(x, y, size, batch_numbers, cfg):

    df = pd.concat([x, y], axis=1)
    X = []
    y = []
    for batch_nr in batch_numbers:
        temp_df = df.loc[df["batch"] == batch_nr]
        if (len(temp_df)) == 0:
            continue
        idx = np.round(np.linspace(0, len(temp_df) - 1, size)).astype(int)
        window = temp_df.iloc[idx, :-2]
        label = temp_df.iloc[0, -1]
        X.append(np.array(window))
        y.append(label - cfg.SEQUENCE.LABEL_OFFSET)

    return np.array(X), to_categorical(np.array(y))
