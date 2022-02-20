from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

PATH = "/media/sebastian/STORAGE_HDD/data/rose_data_pc_2.csv"
COL_TO_DROP = ["class", "batch", "camera_id", "subject_id", "R_id"]
OUTPUT_DIR = Path("generated_dataset")
LIMBS = [
    (1, 2),
    (2, 3),
    (3, 4),
    (1, 5),
    (5, 6),
    (6, 7),
    (1, 0),
    (0, 15),
    (15, 17),
    (0, 16),
    (16, 18),
    (8, 9),
    (9, 10),
    (10, 11),
    (11, 24),
    (11, 22),
    (22, 23),
    (8, 12),
    (12, 13),
    (13, 14),
    (14, 21),
    (14, 19),
    (19, 20),
    (1, 8),
]


def generate():

    features = pd.read_csv(PATH)
    av_angles = read_txt_to_list("/media/sebastian/STORAGE_HDD/data/average_angles.txt")
    av_norm = read_txt_to_list("/media/sebastian/STORAGE_HDD/data/average_norm.txt")
    av_limbs = read_txt_to_list("/media/sebastian/STORAGE_HDD/data/average_limbs.txt")
    features = features.drop(features.columns[0], axis=1)
    print(features)
    # features.replace([np.inf, -np.inf], np.nan, inplace=True)
    # features.interpolate(method="linear", inplace=True, axis=1)
    img_size = 256
    av_limbs = av_limbs * img_size
    frame = 1
    prev_batch = 0
    for index, row in tqdm(features.iterrows(), total=features.shape[0]):
        batch = row["batch"]
        x = row.drop(COL_TO_DROP).astype("float16")

        x = x.values
        x = np.delete(x, np.arange(2, x.size, 3))

        x_per1 = x[50:] * img_size
        x_per2 = x[:50] * img_size

        img = np.zeros([img_size, img_size, 1], dtype=np.uint8)

        x_coords_per1 = []
        it = iter(x_per1)
        joint_coords = tuple(zip(it, it))
        x_coords_per1.append(joint_coords)
        x_coords_per1 = np.array(x_coords_per1[0])

        x_coords_per2 = []
        it = iter(x_per2)
        joint_coords = tuple(zip(it, it))
        x_coords_per2.append(joint_coords)
        x_coords_per2 = np.array(x_coords_per2[0])

        for coord in x_coords_per1:
            if not any(coord):
                continue
            coord = tuple(coord)
            coord = tuple(map(int, coord))
            cv2.circle(img, coord, 2, 127, -1)
        for [start, end] in LIMBS:

            start_point = x_coords_per1[start]
            end_point = x_coords_per1[end]
            if not all(start_point) or not all(end_point):
                continue
            start_point = normalize_coord(start_point)
            end_point = normalize_coord(end_point)
            cv2.line(img, start_point, end_point, 127, 1)
        for coord in x_coords_per2:
            if not any(coord):
                continue
            coord = tuple(coord)
            coord = tuple(map(int, coord))
            cv2.circle(img, coord, 2, 255, -1)
        for [start, end] in LIMBS:

            start_point = x_coords_per2[start]
            end_point = x_coords_per2[end]
            if not all(start_point) or not all(end_point):
                continue
            start_point = normalize_coord(start_point)
            end_point = normalize_coord(end_point)
            cv2.line(img, start_point, end_point, 255, 1)
            ["class", "batch", "camera_id", "subject_id", "R_id"]
        _class, batch, camera_id, subject_id, R_id = (
            row["class"],
            row["batch"],
            row["camera_id"],
            row["subject_id"],
            row["R_id"],
        )
        filename = f"{_class}_{batch}_{camera_id}_{subject_id}_{R_id}_{frame}.jpg"
        frame += 1
        if prev_batch != batch:
            frame = 1

        prev_batch = batch
        cv2.imwrite(str(OUTPUT_DIR / filename), img)


def read_txt_to_list(path: str):
    out = []
    with open(path, "r") as f:
        for line in f.readlines():
            out.append(line.strip("\n"))
    return out


def normalize_coord(coord):
    coord = tuple(coord)
    coord = tuple(map(int, coord))
    return coord


if __name__ == "__main__":
    generate()
