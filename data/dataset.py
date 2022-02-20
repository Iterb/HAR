from pickle import dump

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

import wandb
from sklearn import preprocessing
from utils.make_sequences import create_spaced_sequences, create_spaced_sequences2


class Dataset:
    """Dataset"""

    def __init__(
        self,
        cfg,
    ):
        self.data = pd.read_csv(
            cfg.DATASETS.FULL,
            usecols=["class", "batch", "camera_id", "subject_id", "R_id"],
        )
        # self.data = self.data.drop(self.data.columns[0], axis=1)

        self.scaler = preprocessing.StandardScaler()
        self.imputer = SimpleImputer()
        self.scaler_dist = preprocessing.StandardScaler()
        self.imputer_dist = SimpleImputer()
        self.cfg = cfg

    def _process_for_single_LSTM(self):
        if wandb.config.features_type == 0:
            df = pd.read_csv(self.cfg.DATASETS.FULL, dtype="float32")
            df = df.drop(df.columns[0], axis=1)
        elif wandb.config.features_type == 1:
            features = pd.read_csv(self.cfg.DATASETS.FEATURES_FULL, dtype="float32")
            features = features.drop(features.columns[0], axis=1)
        elif wandb.config.features_type == 2:
            features = pd.read_csv(self.cfg.DATASETS.FEATURES2_FULL, dtype="float16")
            features = features.drop(features.columns[0], axis=1)
        elif wandb.config.features_type == 3:
            features = pd.read_csv(self.cfg.DATASETS.FEATURES3_FULL, dtype="float16")
            features = features.drop(features.columns[0], axis=1)

        print("loaded csvs")
        # print(df)
        df = pd.concat(
            [
                features,
                self.data[["class", "batch", "camera_id", "subject_id", "R_id"]],
            ],
            axis=1,
        )
        del features
        df = df.reset_index(drop=True)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.interpolate(method="linear", inplace=True, axis=1)

        if self.cfg.DATASETS.SPLIT_TYPE == "cs":
            full_data_train = df.loc[
                df["subject_id"].isin(self.cfg.DATASETS.TRAIN_SUBJECTS)
            ]  # .astype("float32")
            full_data_test = df.loc[
                ~df["subject_id"].isin(self.cfg.DATASETS.TRAIN_SUBJECTS)
            ]  # .astype("float32")
        elif self.cfg.DATASETS.SPLIT_TYPE == "cv":
            full_data_train = df.loc[
                df["camera_id"].isin(self.cfg.DATASETS.TRAIN_CAMERAS)
            ]  # .astype("float32")
            full_data_test = df.loc[
                ~df["camera_id"].isin(self.cfg.DATASETS.TRAIN_CAMERAS)
            ]  # .astype("float32")
        del df
        train_indices = np.unique(full_data_train["batch"])
        print("1")
        x_train = full_data_train.drop(
            ["class", "batch", "camera_id", "subject_id", "R_id"], axis=1
        )  # .astype("float32")
        # x_train = x_train.reset_index(drop=True)
        # x_train = x_train.rename(
        #     columns=dict(zip(x_train.columns, range(len(x_train.columns))))
        # ).reset_index(drop=True)
        print(x_train)
        x_train = pd.DataFrame(
            self.imputer.fit_transform(x_train),
            columns=x_train.columns,
            index=x_train.index,
        )
        x_train = pd.DataFrame(
            self.scaler.fit_transform(x_train),
            columns=x_train.columns,
            index=x_train.index,
        )
        x_train = pd.concat([x_train, full_data_train[["batch"]]], axis=1)

        x_test = full_data_test.drop(
            ["class", "batch", "camera_id", "subject_id", "R_id"], axis=1
        )  # .astype("float16")
        # x_test = x_test.reset_index(drop=True)
        x_test = x_test.rename(
            columns=dict(zip(x_test.columns, range(len(x_test.columns))))
        )

        x_test = pd.DataFrame(
            self.imputer.transform(x_test), columns=x_test.columns, index=x_test.index
        )
        x_test = pd.DataFrame(
            self.scaler.transform(x_test), columns=x_test.columns, index=x_test.index
        )
        x_test = pd.concat([x_test, full_data_test[["batch"]]], axis=1)

        y_train = full_data_train[["class"]]
        y_test = full_data_test[["class"]]
        print("processed data")

        dump(
            self.scaler,
            open(
                f"sklearn/{self.cfg.MODEL.ARCH}_F{self.cfg.DATASETS.FEATURES_TYPE}_scaler.pkl",
                "wb",
            ),
        )
        dump(
            self.imputer,
            open(
                f"sklearn/{self.cfg.MODEL.ARCH}_F{self.cfg.DATASETS.FEATURES_TYPE}_imputer.pkl",
                "wb",
            ),
        )

        return x_train, x_test, y_train, y_test, train_indices

    def _process_for_double_LSTM(self):
        if wandb.config.features_type == 1:
            f_p1 = pd.read_csv(
                self.cfg.DATASETS.FEATURES_PER1_D, dtype="float32", header=None
            )
            f_p2 = pd.read_csv(
                self.cfg.DATASETS.FEATURES_PER2_D, dtype="float32", header=None
            )

        elif wandb.config.features_type == 2:
            f_p1 = pd.read_csv(
                self.cfg.DATASETS.FEATURES2_PER1_D, dtype="float32", header=None
            )
            f_p2 = pd.read_csv(
                self.cfg.DATASETS.FEATURES2_PER2_D, dtype="float32", header=None
            )
        elif wandb.config.features_type == 3:
            f_p1 = pd.read_csv(
                self.cfg.DATASETS.FEATURES3_PER1_D, dtype="float32", header=None
            )
            f_p2 = pd.read_csv(
                self.cfg.DATASETS.FEATURES3_PER2_D, dtype="float32", header=None
            )

        f_p1 = f_p1.drop(f_p1.columns[0], axis=1)
        f_p2 = f_p2.drop(f_p2.columns[0], axis=1)
        f_per1 = self._fix_dataframe(f_p1)
        f_per2 = self._fix_dataframe(f_p2)
        if self.cfg.DATASETS.SPLIT_TYPE == "cs":
            full_data_train_per1 = f_per1.loc[
                f_per1["subject_id"].isin(self.cfg.DATASETS.TRAIN_SUBJECTS)
            ]
            full_data_train_per2 = f_per2.loc[
                f_per2["subject_id"].isin(self.cfg.DATASETS.TRAIN_SUBJECTS)
            ]
            full_data_test_per1 = f_per1.loc[
                ~f_per1["subject_id"].isin(self.cfg.DATASETS.TRAIN_SUBJECTS)
            ]
            full_data_test_per2 = f_per2.loc[
                ~f_per2["subject_id"].isin(self.cfg.DATASETS.TRAIN_SUBJECTS)
            ]
        elif self.cfg.DATASETS.SPLIT_TYPE == "cv":
            full_data_train_per1 = f_per1.loc[
                f_per1["camera_id"].isin(self.cfg.DATASETS.TRAIN_CAMERAS)
            ]
            full_data_train_per2 = f_per2.loc[
                f_per2["camera_id"].isin(self.cfg.DATASETS.TRAIN_CAMERAS)
            ]
            full_data_test_per1 = f_per1.loc[
                ~f_per1["camera_id"].isin(self.cfg.DATASETS.TRAIN_CAMERAS)
            ]
            full_data_test_per2 = f_per2.loc[
                ~f_per2["camera_id"].isin(self.cfg.DATASETS.TRAIN_CAMERAS)
            ]

        train_indices = np.unique(full_data_train_per1["batch"])
        x_train_per1 = self.scale_n_impute(
            full_data_train_per1, self.scaler, self.imputer, True
        )

        x_train_per2 = self.scale_n_impute(
            full_data_train_per2, self.scaler, self.imputer, False
        )

        x_test_per1 = self.scale_n_impute(
            full_data_test_per1, self.scaler, self.imputer, False
        )

        x_test_per2 = self.scale_n_impute(
            full_data_test_per2, self.scaler, self.imputer, False
        )

        y_train = full_data_train_per1[["class"]]
        y_test = full_data_test_per1[["class"]]

        dump(
            self.scaler,
            open(
                f"sklearn/{self.cfg.MODEL.ARCH}_F{self.cfg.DATASETS.FEATURES_TYPE}_scaler.pkl",
                "wb",
            ),
        )
        dump(
            self.imputer,
            open(
                f"sklearn/{self.cfg.MODEL.ARCH}_F{self.cfg.DATASETS.FEATURES_TYPE}_imputer.pkl",
                "wb",
            ),
        )

        return (
            x_train_per1,
            x_train_per2,
            x_test_per1,
            x_test_per2,
            y_train,
            y_test,
            train_indices,
        )

    # TODO Rename this here and in `_process_for_double_LSTM`
    def _fix_dataframe(self, arg0):
        result = pd.concat(
            [
                arg0,
                self.data[["class", "batch", "camera_id", "subject_id", "R_id"]],
            ],
            axis=1,
        )

        result = result.reset_index(drop=True)
        result.replace([np.inf, -np.inf], np.nan, inplace=True)
        result.interpolate(method="linear", inplace=True, axis=1)

        return result

    def scale_n_impute(self, data, scaler, imputer, train):
        result = data.drop(
            ["class", "batch", "camera_id", "subject_id", "R_id"], axis=1
        )
        if not train:
            result = pd.DataFrame(
                imputer.transform(result),
                columns=result.columns,
                index=result.index,
            )

            result = pd.DataFrame(
                scaler.transform(result),
                columns=result.columns,
                index=result.index,
            )
        else:
            result = pd.DataFrame(
                imputer.fit_transform(result),
                columns=result.columns,
                index=result.index,
            )

            result = pd.DataFrame(
                scaler.fit_transform(result),
                columns=result.columns,
                index=result.index,
            )

        result = pd.concat([result, data[["batch"]]], axis=1)

        return result

    def _process_for_triple_LSTM(self):

        f_p1 = pd.read_csv(self.cfg.DATASETS.FEATURES_PER1, index_col=[0])
        f_p2 = pd.read_csv(self.cfg.DATASETS.FEATURES_PER2, index_col=[0])
        dist = pd.read_csv(self.cfg.DATASETS.FEATURES_DIST, index_col=[0])

        f_per1 = self._fix_dataframe(f_p1)
        f_per2 = self._fix_dataframe(f_p2)
        distance = self._fix_dataframe(dist)

        if self.cfg.DATASETS.SPLIT_TYPE == "cs":
            full_data_train_per1 = f_per1.loc[
                f_per1["subject_id"].isin(self.cfg.DATASETS.TRAIN_SUBJECTS)
            ]
            full_data_train_per2 = f_per2.loc[
                f_per2["subject_id"].isin(self.cfg.DATASETS.TRAIN_SUBJECTS)
            ]
            full_data_test_per1 = f_per1.loc[
                ~f_per1["subject_id"].isin(self.cfg.DATASETS.TRAIN_SUBJECTS)
            ]
            full_data_test_per2 = f_per2.loc[
                ~f_per2["subject_id"].isin(self.cfg.DATASETS.TRAIN_SUBJECTS)
            ]
            full_data_train_dist = distance.loc[
                distance["subject_id"].isin(self.cfg.DATASETS.TRAIN_SUBJECTS)
            ]
            full_data_test_dist = distance.loc[
                ~distance["subject_id"].isin(self.cfg.DATASETS.TRAIN_CAMERAS)
            ]

        elif self.cfg.DATASETS.SPLIT_TYPE == "cv":
            full_data_train_per1 = f_per1.loc[
                f_per1["camera_id"].isin(self.cfg.DATASETS.TRAIN_CAMERAS)
            ]
            full_data_train_per2 = f_per2.loc[
                f_per2["camera_id"].isin(self.cfg.DATASETS.TRAIN_CAMERAS)
            ]
            full_data_test_per1 = f_per1.loc[
                ~f_per1["camera_id"].isin(self.cfg.DATASETS.TRAIN_CAMERAS)
            ]
            full_data_test_per2 = f_per2.loc[
                ~f_per2["camera_id"].isin(self.cfg.DATASETS.TRAIN_CAMERAS)
            ]
            full_data_train_dist = distance.loc[
                distance["camera_id"].isin(self.cfg.DATASETS.TRAIN_SUBJECTS)
            ]
            full_data_test_dist = distance.loc[
                ~distance["camera_id"].isin(self.cfg.DATASETS.TRAIN_CAMERAS)
            ]

        train_indices = np.unique(full_data_train_per1["batch"])

        x_train_per1 = self.scale_n_impute(
            full_data_train_per1, self.scaler, self.imputer, True
        )

        x_train_per2 = self.scale_n_impute(
            full_data_train_per2, self.scaler, self.imputer, False
        )

        x_test_per1 = self.scale_n_impute(
            full_data_test_per1, self.scaler, self.imputer, False
        )

        x_test_per2 = self.scale_n_impute(
            full_data_test_per2, self.scaler, self.imputer, False
        )
        x_train_dist = self.scale_n_impute(
            full_data_train_dist, self.scaler_dist, self.imputer_dist, True
        )
        x_test_dist = self.scale_n_impute(
            full_data_test_dist, self.scaler_dist, self.imputer_dist, False
        )

        y_train = full_data_train_per1[["class"]]
        y_test = full_data_test_per1[["class"]]

        dump(
            self.scaler,
            open(
                f"sklearn/{self.cfg.MODEL.ARCH}_F{self.cfg.DATASETS.FEATURES_TYPE}_scaler.pkl",
                "wb",
            ),
        )
        dump(
            self.imputer,
            open(
                f"sklearn/{self.cfg.MODEL.ARCH}_F{self.cfg.DATASETS.FEATURES_TYPE}_imputer.pkl",
                "wb",
            ),
        )
        dump(
            self.scaler_dist,
            open(
                f"sklearn/{self.cfg.MODEL.ARCH}_F{self.cfg.DATASETS.FEATURES_TYPE}_scaler_dist.pkl",
                "wb",
            ),
        )
        dump(
            self.imputer_dist,
            open(
                f"sklearn/{self.cfg.MODEL.ARCH}_F{self.cfg.DATASETS.FEATURES_TYPE}_imputer_dist.pkl",
                "wb",
            ),
        )
        return (
            x_train_per1,
            x_train_per2,
            x_train_dist,
            x_test_per1,
            x_test_per2,
            x_test_dist,
            y_train,
            y_test,
            train_indices,
        )

    def _process_for_convnet(self):

        if wandb.config.features_type != 2:
            raise ValueError(
                f"Covnet only uses features 2 and {wandb.config.features_type} were defined in cofig"
            )

        features = pd.read_csv(
            self.cfg.DATASETS.FEATURES2_FULL, dtype="float32", header=None
        )
        features = features.drop(features.columns[0], axis=1)
        df = pd.concat(
            [
                features,
                self.data[["class", "batch", "camera_id", "subject_id", "R_id"]],
            ],
            axis=1,
        )

        print("loaded csvs")

        del features
        df = df.reset_index(drop=True)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.interpolate(method="linear", inplace=True, axis=1)

        # print(df.shape)
        if self.cfg.DATASETS.SPLIT_TYPE == "cs":
            full_data_train = df.loc[
                df["subject_id"].isin(self.cfg.DATASETS.TRAIN_SUBJECTS)
            ].astype("float32")
            full_data_test = df.loc[
                ~df["subject_id"].isin(self.cfg.DATASETS.TRAIN_SUBJECTS)
            ].astype("float32")
        elif self.cfg.DATASETS.SPLIT_TYPE == "cv":
            full_data_train = df.loc[
                df["camera_id"].isin(self.cfg.DATASETS.TRAIN_CAMERAS)
            ].astype("float32")
            full_data_test = df.loc[
                ~df["camera_id"].isin(self.cfg.DATASETS.TRAIN_CAMERAS)
            ].astype("float32")
        del df
        train_indices = np.unique(full_data_train["batch"])
        x_train = self._extracted_from__process_for_convnet_54("1", full_data_train)
        x_train = self.scale_n_impute(full_data_train).drop(["batch"], axis=1)
        img_size = 24
        pad_size = (img_size ** 2 - 450) / 2
        pad_size = 63
        x_train_2D = x_train.apply(
            lambda row: np.pad(row.to_numpy(), (pad_size, pad_size)).reshape(
                img_size, img_size
            ),
            axis=1,
        )

        x_test = self._extracted_from__process_for_convnet_54("3", full_data_test)
        x_test = self.scale_n_impute(full_data_test).drop(["batch"], axis=1)

        x_test_2D = x_test.apply(
            lambda row: np.pad(row.to_numpy(), (pad_size, pad_size)).reshape(
                img_size, img_size
            ),
            axis=1,
        )

        y_train = (
            full_data_train[["class", "batch"]]
            .groupby("batch")["class"]
            .max()
            .to_frame()
            .reset_index()
        )
        y_test = (
            full_data_test[["class", "batch"]]
            .groupby("batch")["class"]
            .max()
            .to_frame()
            .reset_index()
        )
        print("processed data")

        dump(
            self.scaler,
            open(
                f"sklearn/{self.cfg.MODEL.ARCH}_F{self.cfg.DATASETS.FEATURES_TYPE}_scaler.pkl",
                "wb",
            ),
        )
        dump(
            self.imputer,
            open(
                f"sklearn/{self.cfg.MODEL.ARCH}_F{self.cfg.DATASETS.FEATURES_TYPE}_imputer.pkl",
                "wb",
            ),
        )

        return (
            pd.concat([x_train_2D, full_data_train["batch"]], axis=1),
            pd.concat([x_test_2D, full_data_test["batch"]], axis=1),
            y_train,
            y_test,
            train_indices,
        )

    def create_images_from_batches(self, df):
        return df.groupby("batch")[df.columns[0]].apply(list).to_frame().reset_index()

    # TODO Rename this here and in `_process_for_convnet`
    def _extracted_from__process_for_convnet_54(self, arg0, arg1):
        result = arg1.drop(
            ["class", "batch", "camera_id", "subject_id", "R_id"], axis=1
        ).astype("float16")

        result = result.rename(
            columns=dict(zip(result.columns, range(len(result.columns))))
        )

        return result

    def _process_for_skeletons(self):

        features = pd.read_csv(
            self.cfg.DATASETS.FEATURES2_FULL, dtype="float32", header=None
        )
        features = features.drop(features.columns[0], axis=1)
        df = pd.concat(
            [
                features,
                self.data[["class", "batch", "camera_id", "subject_id", "R_id"]],
            ],
            axis=1,
        )

        print("loaded csvs")

        df = pd.concat(
            [
                features,
                self.data[["class", "batch", "camera_id", "subject_id", "R_id"]],
            ],
            axis=1,
        )
        del features
        df = df.reset_index(drop=True)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.interpolate(method="linear", inplace=True, axis=1)

        # print(df.shape)
        if self.cfg.DATASETS.SPLIT_TYPE == "cs":
            full_data_train = df.loc[
                df["subject_id"].isin(self.cfg.DATASETS.TRAIN_SUBJECTS)
            ].astype("float32")
            full_data_test = df.loc[
                ~df["subject_id"].isin(self.cfg.DATASETS.TRAIN_SUBJECTS)
            ].astype("float32")
        elif self.cfg.DATASETS.SPLIT_TYPE == "cv":
            full_data_train = df.loc[
                df["camera_id"].isin(self.cfg.DATASETS.TRAIN_CAMERAS)
            ].astype("float32")
            full_data_test = df.loc[
                ~df["camera_id"].isin(self.cfg.DATASETS.TRAIN_CAMERAS)
            ].astype("float32")
        del df
        train_indices = np.unique(full_data_train["batch"])
        x_train = self._extracted_from__process_for_convnet_54("1", full_data_train)
        x_train = self.scale_n_impute(full_data_train).drop(["batch"], axis=1)
        img_size = 24
        pad_size = (img_size ** 2 - 450) / 2
        pad_size = 63
        x_train_2D = x_train.apply(
            lambda row: np.pad(row.to_numpy(), (pad_size, pad_size)).reshape(
                img_size, img_size
            ),
            axis=1,
        )

        x_test = self._extracted_from__process_for_convnet_54("3", full_data_test)
        x_test = self.scale_n_impute(full_data_test).drop(["batch"], axis=1)

        x_test_2D = x_test.apply(
            lambda row: np.pad(row.to_numpy(), (pad_size, pad_size)).reshape(
                img_size, img_size
            ),
            axis=1,
        )

        y_train = (
            full_data_train[["class", "batch"]]
            .groupby("batch")["class"]
            .max()
            .to_frame()
            .reset_index()
        )
        y_test = (
            full_data_test[["class", "batch"]]
            .groupby("batch")["class"]
            .max()
            .to_frame()
            .reset_index()
        )
        print("processed data")

        dump(
            self.scaler,
            open(
                f"sklearn/{self.cfg.MODEL.ARCH}_F{self.cfg.DATASETS.FEATURES_TYPE}_scaler.pkl",
                "wb",
            ),
        )
        dump(
            self.imputer,
            open(
                f"sklearn/{self.cfg.MODEL.ARCH}_F{self.cfg.DATASETS.FEATURES_TYPE}_imputer.pkl",
                "wb",
            ),
        )

        return (
            pd.concat([x_train_2D, full_data_train["batch"]], axis=1),
            pd.concat([x_test_2D, full_data_test["batch"]], axis=1),
            y_train,
            y_test,
            train_indices,
        )

    def create_imgs_with_skeletons(row):
        x = row.drop(
            ["class", "batch", "camera_id", "subject_id", "R_id"], axis=1
        ).astype("float16")

        avg = x.drop(x.columns[2::3], axis=1)
        # spliting data
        img_size = 128
        x_per1 = avg.drop(avg.columns[50:], axis=1) * img_size
        x_per2 = avg.drop(avg.columns[:50], axis=1) * img_size
        img = np.zeros([img_size, img_size, 1], dtype=np.uint8)

        x_coords_per1 = []
        for x in x_per1.to_numpy():
            it = iter(x)
            joint_coords = list(zip(it, it))
            x_coords_per1.append(joint_coords)
            x_coords_per1 = np.array(x_coords_per1)

        x_coords_per2 = []
        for x in x_per2.to_numpy():
            it = iter(x)
            joint_coords = list(zip(it, it))
            x_coords_per2.append(joint_coords)
            x_coords_per2 = np.array(x_coords_per2)

    def process_data(self):

        all_batches = list(range(self.data["batch"].max()))

        if self.cfg.MODEL.ARCH == "single":
            (
                x_train,
                x_test,
                y_train,
                y_test,
                train_batches,
            ) = self._process_for_single_LSTM()
            X_train_seq, y_train_seq = create_spaced_sequences(
                x_train, y_train, wandb.config.number_of_frames, train_batches, self.cfg
            )
            test_batches = [
                batch for batch in all_batches if batch not in train_batches
            ]
            X_test_seq, y_test_seq = create_spaced_sequences(
                x_test, y_test, wandb.config.number_of_frames, test_batches, self.cfg
            )
            print("created seuences")

            return (X_train_seq, y_train_seq, X_test_seq, y_test_seq)

        if self.cfg.MODEL.ARCH == "double":
            (
                x_train_per1,
                x_train_per2,
                x_test_per1,
                x_test_per2,
                y_train,
                y_test,
                train_batches,
            ) = self._process_for_double_LSTM()
            X_train_seq_per1, y_train_seq = create_spaced_sequences(
                x_train_per1,
                y_train,
                wandb.config.number_of_frames,
                train_batches,
                self.cfg,
            )
            X_train_seq_per2, _ = create_spaced_sequences(
                x_train_per2,
                y_train,
                wandb.config.number_of_frames,
                train_batches,
                self.cfg,
            )
            test_batches = [
                batch for batch in all_batches if batch not in train_batches
            ]
            X_test_seq_per1, y_test_seq = create_spaced_sequences(
                x_test_per1,
                y_test,
                wandb.config.number_of_frames,
                test_batches,
                self.cfg,
            )
            X_test_seq_per2, _ = create_spaced_sequences(
                x_test_per2,
                y_test,
                wandb.config.number_of_frames,
                test_batches,
                self.cfg,
            )
            print("created seuences")

            return (
                X_train_seq_per1,
                X_train_seq_per2,
                y_train_seq,
                X_test_seq_per1,
                X_test_seq_per2,
                y_test_seq,
            )

        if self.cfg.MODEL.ARCH == "triple":
            (
                x_train_per1,
                x_train_per2,
                x_train_dist,
                x_test_per1,
                x_test_per2,
                x_test_dist,
                y_train,
                y_test,
                train_batches,
            ) = self._process_for_triple_LSTM()
            X_train_seq_per1, y_train_seq, X_train_dist_seq = create_spaced_sequences2(
                x_train_per1,
                y_train,
                x_train_dist,
                wandb.config.number_of_frames,
                train_batches,
                self.cfg,
            )
            X_train_seq_per2, _, _ = create_spaced_sequences2(
                x_train_per2,
                y_train,
                x_train_dist,
                wandb.config.number_of_frames,
                train_batches,
                self.cfg,
            )
            test_batches = [
                batch for batch in all_batches if batch not in train_batches
            ]
            X_test_seq_per1, y_test_seq, X_test_dist_seq = create_spaced_sequences2(
                x_test_per1,
                y_test,
                x_test_dist,
                wandb.config.number_of_frames,
                test_batches,
                self.cfg,
            )
            X_test_seq_per2, _, _ = create_spaced_sequences2(
                x_test_per2,
                y_test,
                x_test_dist,
                wandb.config.number_of_frames,
                test_batches,
                self.cfg,
            )

            return (
                X_train_seq_per1,
                X_train_seq_per2,
                X_train_dist_seq,
                X_test_seq_per1,
                X_test_seq_per2,
                X_test_dist_seq,
                y_train_seq,
                y_test_seq,
            )
