import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import cv2
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer

from utils.make_sequences import create_sequences, create_sequences2, create_spaced_sequences

class Dataset:
    """Dataset
    ## TODO
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)
    """

    def __init__(
            self,
            cfg,
    ):
        self.data = pd.read_csv(cfg.DATASETS.FULL)
        self.data = self.data.drop(self.data.columns[0], axis=1)
        self.features = pd.read_csv(cfg.DATASETS.FEATURES)
        self.scaler = preprocessing.StandardScaler()
        self.imputer = SimpleImputer()
        self.cfg = cfg
        print("loaded csvs")

    def _process_for_single_LSTM(self):
        
        df = pd.concat([self.features, self.data[['class','batch',"camera_id", "subject_id", "R_id"]]], axis=1)
        if self.cfg.DATASETS.SPLIT_TYPE == "cs":
            full_data_train = df.loc[df['subject_id'].isin(self.cfg.DATASETS.TRAIN_SUBJECTS)]
            full_data_test = df.loc[~df['subject_id'].isin(self.cfg.DATASETS.TRAIN_SUBJECTS)]
        elif self.cfg.DATASETS.SPLIT_TYPE == "cv":
            full_data_train = df.loc[df['camera_id'].isin(self.cfg.DATASETS.TRAIN_CAMERAS)]
            full_data_test = df.loc[~df['camera_id'].isin(self.cfg.DATASETS.TRAIN_CAMERAS)]

        train_indices = np.unique(full_data_train["batch"])

        x_train = full_data_train.drop(["class", "batch", "camera_id", "subject_id", "R_id"], axis=1)
        x_train = pd.DataFrame(self.imputer.fit_transform(x_train), columns=x_train.columns, index=x_train.index)

        x_train = pd.DataFrame(self.scaler.fit_transform(x_train), columns=x_train.columns, index=x_train.index)
        x_train = pd.concat([x_train, full_data_train[["batch"]]], axis=1)

        x_test = full_data_test.drop(["class", "batch", "camera_id", "subject_id", "R_id"], axis=1)
        x_test = pd.DataFrame(self.imputer.fit_transform(x_test), columns=x_test.columns, index=x_test.index)
        x_test = pd.DataFrame(self.scaler.fit_transform(x_test), columns=x_test.columns, index=x_test.index)
        x_test = pd.concat([x_test, full_data_test[["batch"]]], axis=1)

        y_train = full_data_train[["class"]]
        y_test = full_data_test[["class"]]
        print("processed data")

        return x_train, x_test, y_train, y_test, train_indices

    def create_test_train_sets(self):

        all_batches = [n for n in range(self.data["batch"].max())]
        
        if self.cfg.MODEL.ARCH == 'single':
            x_train, x_test, y_train, y_test, train_batches = self._process_for_single_LSTM()
            X_train_seq, y_train_seq = create_spaced_sequences(x_train, y_train, self.cfg.SEQUENCE.LIN_SIZE, train_batches, self.cfg)
            test_batches = [batch for batch in all_batches if batch not in train_batches]
            X_test_seq, y_test_seq = create_spaced_sequences(x_test, y_test, self.cfg.SEQUENCE.LIN_SIZE, test_batches, self.cfg)
            print("created seuences")

            return X_train_seq, y_train_seq, X_test_seq, y_test_seq


