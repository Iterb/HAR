import itertools
import os
import sys
import timeit
from sys import platform
from typing import Any, List, Type

import cv2
import numpy as np
import pandas as pd
import yacs

from engine.tracker import Tracker
from utils.display import put_interactions_on_video

FEATURES_TYPE_1 = 1
FEATURES_TYPE_2 = 2
FEATURES_TYPE_3 = 3
SINGLE_LSTM = "single"
DOUBLE_LSTM = "double"
TRIPLE_LSTM = "triple"

LIMBS = [
    (1, 2),
    (2, 3),
    (3, 4),
    (1, 5),
    (5, 6),
    (6, 7),
    (0, 1),
    (8, 9),
    (9, 10),
    (10, 11),
    (8, 12),
    (12, 13),
    (13, 14),
]
# direction - away from joint number 1 and then from down up
ANGLES = [
    (8, 1, 2),
    (1, 2, 3),
    (2, 3, 4),
    (8, 1, 5),
    (1, 5, 6),
    (5, 6, 7),
    (2, 1, 0),
    (1, 8, 9),
    (8, 9, 10),
    (9, 10, 11),
    (1, 8, 12),
    (8, 12, 13),
    (12, 13, 14),
]


class pose_tracklet:
    def __init__(self, id):
        self.id = id
        self.keypoints = []


class VideoHander:
    @staticmethod
    def load_video(
        cfg: yacs.config.CfgNode, save_output=False, full_video=False
    ) -> List[Type[pd.DataFrame]]:
        frames_intervals = sample_video(
            cfg.INFER.VIDEO_PATH, cfg.INFER.WINDOW_DURATION_S, cfg.INFER.WINDOW_OFFSET_S
        )
        all_possible_features = []
        if full_video:
            frames_intervals = [[0, np.amax(frames_intervals)]]
        for frame_interval in frames_intervals:
            pose_frame_data = estimate_pose(
                cfg=cfg,
                start_frame=np.min(frame_interval),
                end_frame=np.max(frame_interval),
                save_output=save_output,
            )
            pose_frame_data = np.array(pose_frame_data)
            df_poses_list = create_df_for_each_person(pose_frame_data)
            all_possible_interactions = itertools.combinations(df_poses_list, 2)
            all_possible_features.extend(
                create_features(
                    feature_type=cfg.INFER.FEATURES_TYPE,
                    architecture_type=cfg.INFER.ARCH,
                    person_1_pose=person_pose[0],
                    person_2_pose=person_pose[1],
                )
                for person_pose in all_possible_interactions
            )
        return all_possible_features
    
    @staticmethod
    def save_video(cfg, prediction):
        pose_features = VideoHander.load_video(cfg, save_output=True, full_video=True)
        put_interactions_on_video(
            cfg.INFER.OUTPUT_PATH,
            prediction,
            cfg.INFER.WINDOW_DURATION_S,
            cfg.INFER.WINDOW_OFFSET_S,
            cfg.INFER.OUTPUT_PATH,
        )


def estimate_pose(
    cfg, start_frame: int, end_frame: int, save_output: bool = False
) -> np.array:
    """
    Estimate pose from video.

    Arguments:
    path -- path to input video

    Returns:
    keypoints -- keypoints of estimated pose.
    """
    do_track = cfg.INFER.DO_TRACK
    do_show = cfg.INFER.DO_SHOW

    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(f"{dir_path}/../../python/openpose/Release")
            os.environ["PATH"] = (
                os.environ["PATH"]
                + ";"
                + dir_path
                + "/../../x64/Release;"
                + dir_path
                + "/../../bin;"
            )
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append("/openpose/build/python")
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print(
            "Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?"
        )
        raise e from e

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = {
        "model_folder": "/openpose/models/",
        "face": False,
        "hand": False,
        "keypoint_scale": 3,
        "number_people_max": 2,
    }

    # params["maximize_positives"] = True

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    cap = cv2.VideoCapture(cfg.INFER.VIDEO_PATH)
    if save_output:
        out = cv2.VideoWriter(
            cfg.INFER.OUTPUT_PATH,
            cv2.VideoWriter_fourcc(*"XVID"),
            cap.get(cv2.CAP_PROP_FPS),
            (int(cap.get(3)), int(cap.get(4))),
        )

    pose_frame_data = []
    current_frame_number = -1
    if do_track:
        width = int(cap.get(3))  # float `width`
        height = int(cap.get(4))  # float `height`
        tracker = Tracker(cfg, width, height)
    print(start_frame, end_frame)
    starttime = timeit.default_timer()
    while True:
        # Capture frame-by-frame
        current_frame_number += 1
        ret, frame = cap.read()
        if current_frame_number < start_frame:
            continue
        if current_frame_number > end_frame:
            cv2.destroyAllWindows()
            break

        datum = op.Datum()
        datum.cvInputData = frame
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        currentFrame = datum.cvOutputData
        if do_track:
            tracks, currentFrame = tracker.track(datum)
        if do_show:
            if do_track:
                for track in tracks:
                    color = None
                    color = (255, 255, 255)
                    bbox = track.to_tlbr()
                    # print(bbox)
                    cv2.rectangle(
                        currentFrame,
                        (int(bbox[0]), int(bbox[1])),
                        (int(bbox[2]), int(bbox[3])),
                        color,
                        2,
                    )
                    cv2.putText(
                        currentFrame,
                        "id%s" % (track.track_id),
                        (
                            int(bbox[0]) + int((int(bbox[2]) - int(bbox[0])) / 2) - 5,
                            int(bbox[1]) + 20,
                        ),
                        0,
                        5e-3 * 100,
                        (255, 255, 255),
                        1,
                    )
            cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", currentFrame)
        if datum.poseKeypoints.shape[0] >= 2:  ## FIX THAT
            two_people_pose = datum.poseKeypoints[:2]  ## FIX THAT
            pose_frame_data.append(two_people_pose)
        if save_output:
            out.write(currentFrame.astype("uint8"))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        # print()
    cap.release()
    if save_output:
        out.release()
    time_delta = timeit.default_timer() - starttime
    print(
        f"The time difference is : {time_delta} for {end_frame- start_frame} frames which eq to {(end_frame- start_frame)/time_delta} FPS"
    )

    return pose_frame_data


def sample_video(path: str, duration_in_sec: int, interval_in_sec: int):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_in_frames = duration_in_sec * fps
    interval_in_frames = interval_in_sec * fps
    # for interval_frames in enumerate(frame_number)
    all_frames = np.arange(frame_number, dtype=np.int32)
    max_intervals = ((frame_number - duration_in_frames) // interval_in_frames) + 1
    return [
        all_frames[
            int(interval_in_frames) * i : int(interval_in_frames) * i
            + int(duration_in_frames)
        ]
        for i in range(int(max_intervals))
    ]


def create_df_for_each_person(pose_data: np.array) -> List[Type[pd.DataFrame]]:
    # we have array[frame][person][keypoint-nr][coords] we want [person][frame][keypoint-nr][coords]
    pose_data_T = pose_data.transpose(1, 0, 2, 3)
    df_list = []
    for person in pose_data_T:
        # from [frame][keypoint-nr][coords] to [frame][value]
        person = person.reshape(-1, 75)
        df = pd.DataFrame(person)
        df_list.append(df)

    return df_list


def create_features(
    feature_type: int,
    architecture_type: str,
    person_1_pose: pd.DataFrame,
    person_2_pose: pd.DataFrame,
) -> pd.DataFrame:
    x = pd.concat([person_1_pose, person_2_pose], axis=1)
    x.columns = range(x.shape[1])
    return normalize(x, feature_type, architecture_type)


def calculate_stacked_preditions(window_duration: int, window_offset: int, preds: list):
    number_of_classes = 11
    max_len = window_duration + window_offset * (len(preds) - 1)

    stacked_predictions = np.zeros([max_len, number_of_classes])
    average_stacked_predictions = np.zeros([max_len, number_of_classes])
    divider = np.zeros([max_len])
    for dx, prediction in enumerate(preds):
        for s in range(window_duration):
            stacked_predictions[dx * window_offset + s] += np.array(prediction).reshape(
                11
            )
            divider[dx * window_offset + s] += 1

    average_stacked_predictions = np.divide(stacked_predictions, divider.reshape(-1, 1))

    return average_stacked_predictions


def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(cosine_angle)) / 180


def calculate_average_limb_lengths(x_coords):
    x_avg_lengths = [[] for i in range(len(LIMBS))]
    norms = []
    # x_avg_lengths[0].append(1)
    for frame in x_coords:
        if (np.any(frame[1] == 0)) or (np.any(frame[8] == 0)):
            continue
        norm = np.linalg.norm(frame[1] - frame[8])
        norms.append(norm)
        l0 = 1
        for i, limb in enumerate(LIMBS):
            if (np.all(frame[limb[0]] != 0)) and (np.all(frame[limb[1]] != 0)):
                l = np.linalg.norm(frame[limb[0]] - frame[limb[1]]) / norm
                x_avg_lengths[i].append(l)

    limb_lengths_average = [sum(col) / len(col) for col in x_avg_lengths]
    norm_average = sum(norms) / len(norms)
    return limb_lengths_average, norm_average


def calculate_average_angles(x_coords):
    x_avg_angles = [[] for _ in range(len(ANGLES))]
    for frame in x_coords:
        for i, angle in enumerate(ANGLES):
            if (
                np.all(frame[angle[0]] != 0)
                and np.all(frame[angle[1]] != 0)
                and np.all(frame[angle[2]] != 0)
            ):

                a = calculate_angle(frame[angle[0]], frame[angle[1]], frame[angle[2]])
                x_avg_angles[i].append(a)
    return [sum(col) / len(col) for col in x_avg_angles]


def calculate_limb_lengths(x_coords, average_LIMBS, average_norm):
    x_lengths = []
    for frame in x_coords:
        x_person = []
        if (np.any(frame[1] == 0)) or (np.any(frame[8] == 0)):
            norm = 1  # average_norm
        else:
            norm = np.linalg.norm(frame[1] - frame[8])

        # x_person.append(1)
        for limb in LIMBS:
            if (np.any(frame[limb[0]] == 0)) or (np.any(frame[limb[1]] == 0)):
                l = 1  # average_LIMBS[i]
            else:
                try:
                    l = np.linalg.norm(frame[limb[0]] - frame[limb[1]]) / norm
                except:
                    l = 1
            x_person.append(l)
        x_lengths.append(x_person)
    return x_lengths


def angle_between_vectors(vector_1, vector_2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    return np.arccos(dot_product)


def calculate_angles(x_coords, average_ang):
    x_angles = []
    for frame in x_coords:
        x_person = []
        for angle in ANGLES:
            if (
                np.any(frame[angle[0]] == 0)
                or np.any(frame[angle[1]] == 0)
                or np.any(frame[angle[2]] == 0)
            ):
                a = 1  # average_ang[i]
            else:
                a = calculate_angle(frame[angle[0]], frame[angle[1]], frame[angle[2]])
            x_person.append(a)
        x_angles.append(x_person)
    return x_angles


def calculate_distance(x_coords_1, x_coords_2, average_norm, distance_type="2P"):
    x_distance = []

    for frame1, frame2 in zip(x_coords_1, x_coords_2):
        if (np.any(frame1[1] == 0)) or (np.any(frame1[8] == 0)):
            norm = average_norm
        else:
            norm = np.linalg.norm(frame1[1] - frame1[8])
        if distance_type == "2P":
            if (np.any(frame1[1] == 0)) or (np.any(frame2[1] == 0)):
                d = np.linalg.norm(frame1[0] - frame2[0]) / norm
            else:
                d = np.linalg.norm(frame1[1] - frame2[1]) / norm
            x_distance.append(d)
        elif distance_type == "25P":
            x_person = []
            moving_avg_d = 1
            for i in range(25):
                if np.any(frame1[i] == 0) or np.any(frame2[i] == 0):
                    moving_avg_d = (sum(x_person) + moving_avg_d) / (i + 1)
                    d = moving_avg_d
                else:
                    d = np.linalg.norm(frame1[i] - frame2[i]) / norm
                x_person.append(d)
            x_distance.append(x_person)
        elif distance_type == "cross_dist":
            x_person = []
            moving_avg_d = 1
            for i in range(15):
                if np.any(frame1[i] == 0) or np.any(frame2[1] == 0):
                    moving_avg_d = (sum(x_person) + moving_avg_d) / (i + 1)
                    d = moving_avg_d
                else:
                    d = np.linalg.norm(frame1[i] - frame2[1]) / norm
                x_person.append(d)
            x_distance.append(x_person)
        elif distance_type == "ffa":
            x_person = [[0 for _ in range(15)] for _ in range(15)]
            moving_avg_d = 1
            for i in range(15):
                for j in range(15):
                    if np.any(frame1[i] == 0) or np.any(frame2[j] == 0):
                        moving_avg_d = (sum(x_person[i]) + moving_avg_d) / (j + 1)
                        d = moving_avg_d
                    else:
                        d = np.linalg.norm(frame1[i] - frame2[j]) / norm
                    x_person[i][j] = d
            x_distance.append(x_person)
    return x_distance


def polar_coords(x_coords_1, x_coords_2, average_norm):
    results = []
    for frame1, frame2 in zip(x_coords_1, x_coords_2):
        if (np.any(frame1[1] == 0)) or (np.any(frame1[8] == 0)):
            norm = average_norm
        else:
            norm = np.linalg.norm(frame1[1] - frame1[8])

        if (np.any(frame1[1] == 0)) or (np.any(frame2[1] == 0)):
            norm_vector = (1, 0)
        else:
            norm_vector = frame1[1] - frame2[1]

        x_person_dist = [[0 for x in range(15)] for y in range(15)]
        x_person_angle = [[0 for x in range(15)] for y in range(15)]
        moving_avg_d = 1
        for i in range(15):
            for j in range(15):
                moving_avg_d = (sum(x_person_dist[i]) + moving_avg_d) / (j + 1)
                if np.any(frame1[i] == 0) or np.any(frame2[j] == 0):
                    d = moving_avg_d
                    a = 1
                else:
                    d = np.linalg.norm(frame1[i] - frame2[j]) / norm
                    vector = frame1[i] - frame2[j]
                    a = angle_between_vectors(norm_vector, vector)
                x_person_dist[i][j] = d
                x_person_angle[i][j] = a

        result = np.array(x_person_dist + x_person_angle).flatten()
        results.append(result)
    return results


def polar_coords2(x_coords_1, x_coords_2, average_norm):
    results = []
    for frame1, frame2 in zip(x_coords_1, x_coords_2):
        if (np.any(frame1[1] == 0)) or (np.any(frame1[8] == 0)):
            norm = average_norm
            norm_vector = (1, 0)
            middle_point = (0.5, 0.5)
        else:
            norm = np.linalg.norm(frame1[1] - frame1[8])
            norm_vector = frame1[1] - frame2[1]
            middle_point = (frame1[1] + frame2[1]) / 2

        x_person_dist = [0 for y in range(15)]
        x_person_angle = [0 for y in range(15)]
        moving_avg_d = 1
        for i in range(15):
            moving_avg_d = (sum(x_person_dist) + moving_avg_d) / (i + 1)
            if np.any(frame2[i] == 0):
                d = moving_avg_d
                a = 1
            else:
                try:
                    d = np.linalg.norm(middle_point - frame2[i]) / norm
                except TypeError:
                    d = moving_avg_d
                vector = middle_point - frame2[i]
                a = angle_between_vectors(norm_vector, vector)
            x_person_dist[i] = d
            x_person_angle[i] = a

        result = np.array(x_person_dist + x_person_angle).flatten()
        results.append(result)

    return results


def normalize(x, feature_type=FEATURES_TYPE_1, architecture_type=SINGLE_LSTM):
    # maybe add minmax scaling
    # removing c values
    av_angles = read_txt_to_list("/media/sebastian/STORAGE_HDD/data/average_angles.txt")
    av_norm = read_txt_to_list("/media/sebastian/STORAGE_HDD/data/average_norm.txt")
    av_limbs = read_txt_to_list("/media/sebastian/STORAGE_HDD/data/average_limbs.txt")
    x = x.drop(x.columns[2::3], axis=1)
    # spliting data
    x_per1 = x.drop(x.columns[50:], axis=1)
    x_per2 = x.drop(x.columns[:50], axis=1)

    x_coords_per1 = _extracted_from_normalize_12(x_per1)
    x_coords_per2 = _extracted_from_normalize_12(x_per2)
    # Calculating new features
    if feature_type not in [FEATURES_TYPE_2, FEATURES_TYPE_3]:
        x_angles_per1 = calculate_angles(x_coords_per1, av_angles)
        x_angles_per2 = calculate_angles(x_coords_per2, av_angles)
        x_limbs_per1 = calculate_limb_lengths(x_coords_per1, av_limbs, av_norm)
        x_limbs_per2 = calculate_limb_lengths(x_coords_per2, av_limbs, av_norm)

    # Feature type 1 - angles and lengths for both persons concatted into 1 big df
    if feature_type == FEATURES_TYPE_1 and architecture_type == SINGLE_LSTM:
        x_distances = calculate_distance(
            x_coords_per1, x_coords_per2, av_norm, distance_type="25P"
        )
        x_angles_per1_df = pd.DataFrame(x_angles_per1)
        x_angles_per2_df = pd.DataFrame(x_angles_per2)
        x_limbs_per1_df = pd.DataFrame(x_limbs_per1)
        x_limbs_per2_df = pd.DataFrame(x_limbs_per2)
        x_distances_df = pd.DataFrame(x_distances)
        x_features_typeP = pd.concat(
            [
                x_angles_per1_df,
                x_angles_per2_df,
                x_limbs_per1_df,
                x_limbs_per2_df,
                x_distances_df,
            ],
            axis=1,
        )
        x_features_typeP.columns = range(x_features_typeP.shape[1])

        return x_features_typeP

    elif feature_type == FEATURES_TYPE_2 and architecture_type == TRIPLE_LSTM:

        x_distances = calculate_distance(
            x_coords_per1, x_coords_per2, av_norm, distance_type="25P"
        )
        x_angles_per1_df = pd.DataFrame(x_angles_per1)
        x_angles_per2_df = pd.DataFrame(x_angles_per2)
        x_limbs_per1_df = pd.DataFrame(x_limbs_per1)
        x_limbs_per2_df = pd.DataFrame(x_limbs_per2)
        x_distances_df = pd.DataFrame(x_distances)

        x_features_typeV_per1 = pd.concat([x_angles_per1_df, x_limbs_per1_df], axis=1)
        x_features_typeV_per1.columns = range(x_features_typeV_per1.shape[1])
        x_features_typeV_per2 = pd.concat([x_angles_per2_df, x_limbs_per2_df], axis=1)
        x_features_typeV_per2.columns = range(x_features_typeV_per2.shape[1])
        x_distances_df.columns = range(x_distances_df.shape[1])

        return (x_features_typeV_per1, x_features_typeV_per2, x_distances_df)

    elif feature_type == FEATURES_TYPE_3 and architecture_type == DOUBLE_LSTM:
        x_distance_per1 = calculate_distance(
            x_coords_per1, x_coords_per2, av_norm, distance_type="cross_dist"
        )
        x_distance_per2 = calculate_distance(
            x_coords_per2, x_coords_per1, av_norm, distance_type="cross_dist"
        )
        x_angles_per1_df = pd.DataFrame(x_angles_per1)
        x_angles_per2_df = pd.DataFrame(x_angles_per2)
        x_limbs_per1_df = pd.DataFrame(x_limbs_per1)
        x_limbs_per2_df = pd.DataFrame(x_limbs_per2)
        x_distance_per1_df = pd.DataFrame(x_distance_per1)
        x_distance_per2_df = pd.DataFrame(x_distance_per2)

        x_features_typeD_per1 = pd.concat(
            [x_angles_per1_df, x_limbs_per1_df, x_distance_per1_df], axis=1
        )
        x_features_typeD_per1.columns = range(x_features_typeD_per1.shape[1])
        x_features_typeD_per2 = pd.concat(
            [x_angles_per2_df, x_limbs_per2_df, x_distance_per2_df], axis=1
        )
        x_features_typeD_per2.columns = range(x_features_typeD_per2.shape[1])

        return (x_features_typeD_per1, x_features_typeD_per2)

    elif feature_type == FEATURES_TYPE_2 and architecture_type == SINGLE_LSTM:

        x_polar_coords1 = polar_coords(x_coords_per1, x_coords_per2, av_norm)
        x_polar_coords2 = polar_coords(x_coords_per2, x_coords_per1, av_norm)
        x_polar_coords1_df = pd.DataFrame(x_polar_coords1)
        x_polar_coords2_df = pd.DataFrame(x_polar_coords2)
        return _extracted_from_normalize_112(x_polar_coords1_df, x_polar_coords2_df)
    elif feature_type == FEATURES_TYPE_2 and architecture_type == DOUBLE_LSTM:

        x_polar_coords1 = polar_coords(x_coords_per1, x_coords_per2, av_norm)
        x_polar_coords2 = polar_coords(x_coords_per2, x_coords_per1, av_norm)
        x_polar_coords1_df = pd.DataFrame(x_polar_coords1)
        x_polar_coords2_df = pd.DataFrame(x_polar_coords2)
        return (x_polar_coords1_df, x_polar_coords2_df)

    elif feature_type == FEATURES_TYPE_3 and architecture_type == SINGLE_LSTM:
        x_polar_coords1 = polar_coords2(x_coords_per1, x_coords_per2, av_norm)
        x_polar_coords2 = polar_coords2(x_coords_per2, x_coords_per1, av_norm)
        x_polar_coords1_df = pd.DataFrame(x_polar_coords1)
        x_polar_coords2_df = pd.DataFrame(x_polar_coords2)
        return _extracted_from_normalize_112(x_polar_coords1_df, x_polar_coords2_df)
    elif feature_type == FEATURES_TYPE_3 and architecture_type == TRIPLE_LSTM:
        x_polar_coords1 = polar_coords2(x_coords_per1, x_coords_per2, av_norm)
        x_polar_coords2 = polar_coords2(x_coords_per2, x_coords_per1, av_norm)
        x_polar_coords1_df = pd.DataFrame(x_polar_coords1)
        x_polar_coords2_df = pd.DataFrame(x_polar_coords2)
        return (x_polar_coords1_df, x_polar_coords2_df)


def _extracted_from_normalize_12(arg0):
    result = []
    for x in arg0.to_numpy():
        it = iter(x)
        joint_coords = list(zip(it, it))
        result.append(joint_coords)

    result = np.array(result)

    return result


def _extracted_from_normalize_112(x_polar_coords1_df, x_polar_coords2_df):
    x = pd.concat([x_polar_coords1_df, x_polar_coords2_df], axis=1)
    x.columns = range(x.shape[1])
    return x


def read_txt_to_list(path: str) -> List[Type[Any]]:
    out = []
    with open(path, "r") as f:
        for line in f.readlines():
            out.append(line.strip("\n"))
    return out
