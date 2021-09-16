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
        
        self.scaler = preprocessing.StandardScaler()
        self.imputer = SimpleImputer()
        self.cfg = cfg
        print("loaded csvs")

    def _process_for_single_LSTM(self):
        features = pd.read_csv(self.cfg.DATASETS.FEATURES_FULL)
        df = pd.concat([features, self.data[['class','batch',"camera_id", "subject_id", "R_id"]]], axis=1)
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
        x_test = pd.DataFrame(self.imputer.transform(x_test), columns=x_test.columns, index=x_test.index) #prob only transform
        x_test = pd.DataFrame(self.scaler.transform(x_test), columns=x_test.columns, index=x_test.index)
        x_test = pd.concat([x_test, full_data_test[["batch"]]], axis=1)

        y_train = full_data_train[["class"]]
        y_test = full_data_test[["class"]]
        print("processed data")

        return x_train, x_test, y_train, y_test, train_indices
    
    def _process_for_double_LSTM(self):
        f_p1 = pd.read_csv(self.cfg.DATASETS.FEATURES2_PER1)
        f_p2 = pd.read_csv(self.cfg.DATASETS.FEATURES2_PER2)
        f_per1 = pd.concat([f_p1, self.data[['class','batch',"camera_id", "subject_id", "R_id"]]], axis=1)
        f_per2 = pd.concat([f_p2, self.data[['class','batch',"camera_id", "subject_id", "R_id"]]], axis=1)

        if self.cfg.DATASETS.SPLIT_TYPE == "cs":
            full_data_train_per1 = f_per1.loc[f_per1['subject_id'].isin(self.cfg.DATASETS.TRAIN_SUBJECTS)]
            full_data_train_per2 = f_per2.loc[f_per2['subject_id'].isin(self.cfg.DATASETS.TRAIN_SUBJECTS)]
            full_data_test_per1  = f_per1.loc[~f_per1['subject_id'].isin(self.cfg.DATASETS.TRAIN_SUBJECTS)]
            full_data_test_per2 = f_per2.loc[~f_per2['subject_id'].isin(self.cfg.DATASETS.TRAIN_SUBJECTS)]
        elif self.cfg.DATASETS.SPLIT_TYPE == "cv":
            full_data_train_per1 = f_per1.loc[f_per1['camera_id'].isin(self.cfg.DATASETS.TRAIN_CAMERAS)]
            full_data_train_per2 = f_per2.loc[f_per2['camera_id'].isin(self.cfg.DATASETS.TRAIN_CAMERAS)]
            full_data_test_per1  = f_per1.loc[~f_per1['camera_id'].isin(self.cfg.DATASETS.TRAIN_CAMERAS)]
            full_data_test_per2 = f_per2.loc[~f_per2['camera_id'].isin(self.cfg.DATASETS.TRAIN_CAMERAS)]

        train_indices = np.unique(full_data_train_per1["batch"])

        x_train_per1 = full_data_train_per1.drop(["class", "batch", "camera_id", "subject_id", "R_id"], axis=1)
        x_train_per1 = pd.DataFrame(self.imputer.fit_transform(x_train_per1), columns=x_train_per1.columns, index=x_train_per1.index)
        x_train_per1 = pd.DataFrame(self.scaler.fit_transform(x_train_per1), columns=x_train_per1.columns, index=x_train_per1.index)
        x_train_per1 = pd.concat([x_train_per1, full_data_train_per1[["batch"]]], axis=1)

        x_train_per2 = full_data_train_per2.drop(["class", "batch", "camera_id", "subject_id", "R_id"], axis=1)
        x_train_per2 = pd.DataFrame(self.imputer.fit_transform(x_train_per2), columns=x_train_per2.columns, index=x_train_per2.index)
        x_train_per2 = pd.DataFrame(self.scaler.fit_transform(x_train_per2), columns=x_train_per2.columns, index=x_train_per2.index)
        x_train_per2 = pd.concat([x_train_per2, full_data_train_per2[["batch"]]], axis=1)

        x_test_per1 = full_data_test_per1.drop(["class", "batch", "camera_id", "subject_id", "R_id"], axis=1)
        x_test_per1 = pd.DataFrame(self.imputer.fit_transform(x_test_per1), columns=x_test_per1.columns, index=x_test_per1.index)
        x_test_per1 = pd.DataFrame(self.scaler.fit_transform(x_test_per1), columns=x_test_per1.columns, index=x_test_per1.index)
        x_test_per1 = pd.concat([x_test_per1, full_data_test_per1[["batch"]]], axis=1)

        x_test_per2 = full_data_test_per2.drop(["class", "batch", "camera_id", "subject_id", "R_id"], axis=1)
        x_test_per2 = pd.DataFrame(self.imputer.fit_transform(x_test_per2), columns=x_test_per2.columns, index=x_test_per2.index)
        x_test_per2 = pd.DataFrame(self.scaler.fit_transform(x_test_per2), columns=x_test_per2.columns, index=x_test_per2.index)
        x_test_per2 = pd.concat([x_test_per2, full_data_test_per2[["batch"]]], axis=1)


        y_train = full_data_train_per1[["class"]]
        y_test = full_data_test_per1[["class"]]

        return x_train_per1, x_train_per2, x_test_per1, x_test_per2, y_train, y_test, train_indices

    def create_test_train_sets(self):

        all_batches = [n for n in range(self.data["batch"].max())]
        
        if self.cfg.MODEL.ARCH == 'single':
            x_train, x_test, y_train, y_test, train_batches = self._process_for_single_LSTM()
            X_train_seq, y_train_seq = create_spaced_sequences(x_train, y_train, self.cfg.SEQUENCE.LIN_SIZE, train_batches, self.cfg)
            test_batches = [batch for batch in all_batches if batch not in train_batches]
            X_test_seq, y_test_seq = create_spaced_sequences(x_test, y_test, self.cfg.SEQUENCE.LIN_SIZE, test_batches, self.cfg)
            print("created seuences")

            return (X_train_seq, y_train_seq, X_test_seq, y_test_seq)

        if self.cfg.MODEL.ARCH == 'double':
            x_train_per1, x_train_per2, x_test_per1, x_test_per2, y_train, y_test, train_batches = self._process_for_double_LSTM()
            X_train_seq_per1, y_train_seq  = create_spaced_sequences(x_train_per1, y_train, self.cfg.SEQUENCE.LIN_SIZE, train_batches, self.cfg)
            X_train_seq_per2, _ = create_spaced_sequences(x_train_per2, y_train, self.cfg.SEQUENCE.LIN_SIZE, train_batches, self.cfg)
            test_batches = [batch for batch in all_batches if batch not in train_batches]
            X_test_seq_per1, y_test_seq  = create_spaced_sequences(x_test_per1, y_test, self.cfg.SEQUENCE.LIN_SIZE, train_batches, self.cfg)
            X_test_seq_per2, _ = create_spaced_sequences(x_test_per2, y_test, self.cfg.SEQUENCE.LIN_SIZE, train_batches, self.cfg)
            print("created seuences")

            return (X_train_seq_per1, X_train_seq_per2, y_train_seq, X_test_seq_per1, X_test_seq_per2, y_test_seq)


