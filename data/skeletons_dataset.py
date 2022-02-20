from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Literal

import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from tensorflow import keras


class SkeletonDataset:
    def __init__(
        self,
        linspace_size: int,
        imgs_root_dir: str,
        train_instances: List[int],
        train: bool = True,
        data_split: Literal["cv", "cs"] = "cv",
    ):
        self.linspace_size = linspace_size
        self.image_names = Path(imgs_root_dir).glob("*.jpg")
        self.data = pd.DataFrame(self.image_names, columns=["name"])
        self.data = (
            self.data.pipe(self.get_batch_nr)
            .pipe(self.get_frame_nr)
            .pipe(self.get_cam_id)
            .pipe(self.get_subject_id)
            .pipe(self.get_class_id)
            .pipe(
                self.get_train_test_examples,
                train_instances=train_instances,
                data_split=data_split,
                train=train,
            )
            .sort_values(["batch", "frame"])
            .groupby(["batch", "class"])["name"]
            .apply(list)
            .to_frame()
            .reset_index()
        )
        self.image_paths = self.data["name"].to_numpy()
        self.targets = self.data["class"].to_numpy()

    @staticmethod
    def get_batch_nr(df: pd.DataFrame) -> pd.DataFrame:
        """{_class}_{batch}_{camera_id}_{subject_id}_{R_id}_{frame}.jpg"""
        df["batch"] = df["name"].map(
            lambda name: int(float(str(Path(name).stem).split("_")[1]))
        )
        return df

    @staticmethod
    def get_frame_nr(df: pd.DataFrame) -> pd.DataFrame:
        df["frame"] = df["name"].map(
            lambda name: int(float(str(Path(name).stem).split("_")[-1]))
        )
        return df

    @staticmethod
    def get_cam_id(df: pd.DataFrame) -> pd.DataFrame:
        df["camera_id"] = df["name"].map(
            lambda name: int(float(str(Path(name).stem).split("_")[2]))
        )
        return df

    @staticmethod
    def get_subject_id(df: pd.DataFrame) -> pd.DataFrame:
        df["subject_id"] = df["name"].map(
            lambda name: int(float(str(Path(name).stem).split("_")[3]))
        )
        return df

    @staticmethod
    def get_class_id(df: pd.DataFrame) -> pd.DataFrame:
        df["class"] = df["name"].map(
            lambda name: int(float(str(Path(name).stem).split("_")[0]))
        )
        return df

    @staticmethod
    def get_train_test_examples(
        df: pd.DataFrame,
        train_instances: List[int],
        data_split: Literal["cv", "cs"],
        train: bool = True,
    ) -> pd.DataFrame:
        if data_split == "cs":
            if train:
                df = df[df["subject_id"].isin(train_instances)]
            else:
                df = df[~df["subject_id"].isin(train_instances)]

        elif data_split == "cv":
            if train:
                df = df[df["camera_id"].isin(train_instances)]
            else:
                df = df[~df["camera_id"].isin(train_instances)]
        return df
    
    @staticmethod
    def create_volume(image_paths: np.ndarray):
        return np.array(
            [
                cv2.resize
                (
                    cv2.imread(str(image_path), -1),
                    (64, 64)
                ) for image_path in image_paths
            ]  
        )
            
            
    def __getitem__(self, i):

        # read data

        image_paths =np.array(self.image_paths[i])
        indices = np.linspace(0, len(image_paths) - 1, self.linspace_size, dtype=np.int)
        image_paths_norm_len = image_paths[indices]
        image_volume = self.create_volume(image_paths_norm_len)
        target = self.targets[i]
        target = tf.keras.utils.to_categorical(target, 11)
        return image_volume.astype("float32"), target

    def __len__(self):
        return len(self.targets)


def default_collate_fn(samples):
    X = np.array([sample[0] for sample in samples])
    Y = np.array([sample[1] for sample in samples])

    return X, Y
