from typing import Any, List, Type
import sys
import os
from sys import platform
import argparse
import itertools

import numpy as np
import cv2
import pandas as pd
import yacs
FEATURES_TYPE_1 = 1
FEATURES_TYPE_2 = 2
FEATURES_TYPE_3 = 3
SINGLE_LSTM = 'single'
DOUBLE_LSTM = 'double'
TRIPLE_LSTM = 'triple'

LIMBS = [(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),(0,1),(8,9),(9,10),(10,11),(8,12),(12,13),(13,14)]
#direction - away from joint number 1 and then from down up
ANGLES = [(8,1,2),(1,2,3),(2,3,4),(8,1,5),(1,5,6),(5,6,7),(2,1,0),
          (1,8,9),(8,9,10),(9,10,11),(1,8,12),(8,12,13),(12,13,14)]

def extract_pose_features_from_video(cfg: yacs.config.CfgNode) -> List[Type[pd.DataFrame]]:
    print(type(cfg))
    pose_frame_data = process_video(cfg.INFER.VIDEO_PATH)
    pose_frame_data = np.array(pose_frame_data)
    df_poses_list = create_df_for_each_person(pose_frame_data)
    all_possible_interactions = itertools.combinations(df_poses_list, 2)
    print(all_possible_interactions)
    all_possible_features = []
    for person_pose in all_possible_interactions:
        print(person_pose[1])
        all_possible_features.append(create_features(feature_type=cfg.INFER.FEATURES_TYPE,
                                                     architecture_type=cfg.INFER.ARCH,
                                                     person_1_pose=person_pose[0],
                                                     person_2_pose=person_pose[1]))
    return all_possible_features
def process_video(path: str) -> np.array:
    """
    Estimate pose from video.

    Arguments:
    path -- path to input video

    Returns:
    keypoints -- keypoints of estimated pose.
    """
    try:
        # Import Openpose (Windows/Ubuntu/OSX)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        try:
            # Windows Import
            if platform == "win32":
                # Change these variables to point to the correct folder (Release/x64 etc.)
                sys.path.append(dir_path + '/../../python/openpose/Release');
                os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
                import pyopenpose as op
            else:
                # Change these variables to point to the correct folder (Release/x64 etc.)
                sys.path.append('/openpose/build/python');
                # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
                # sys.path.append('/usr/local/python')
                from openpose import pyopenpose as op
        except ImportError as e:
            print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
            raise e

        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        params = dict()
        params["model_folder"] = "/openpose/models/"
        params["face"] = False
        params["hand"] = False
        params["keypoint_scale"] = 3

        # Starting OpenPose
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()
        cap = cv2.VideoCapture(path)

        pose_frame_data = []

        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()

            datum = op.Datum()
            datum.cvInputData = frame
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))

            print("Body keypoints: \n" + str(datum.poseKeypoints))
            print(datum.poseKeypoints.shape)
            cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
            if datum.poseKeypoints.shape[0] == 2: ## FIX THAT
                pose_frame_data.append(datum.poseKeypoints)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

    except Exception as e:
        print(e)

    return pose_frame_data


def create_df_for_each_person(pose_data: np.array) -> List[Type[pd.DataFrame]]:
    #we have array[frame][person][keypoint-nr][coords] we want [person][frame][keypoint-nr][coords] 
    pose_data_T = pose_data.transpose(1,0,2,3)
    df_list = []
    for person in pose_data_T:
        # from [frame][keypoint-nr][coords] to [frame][value] 
        person = person.reshape(-1, 75)
        df = pd.DataFrame(person)
        df_list.append(df)

    return df_list

def create_features(feature_type: int, architecture_type: str, person_1_pose: pd.DataFrame, person_2_pose: pd.DataFrame) -> pd.DataFrame:
    x = pd.concat([person_1_pose, person_2_pose], axis=1)
    x.columns = range(0,x.shape[1])
    features = normalize(x, feature_type, architecture_type)

    return features

def calculate_angle(a, b, c):
  ba = a - b
  bc = c - b
  cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
  return (np.degrees(np.arccos(cosine_angle)) / 180)

def calculate_average_limb_lengths(x_coords):
  x_avg_lengths =[[] for i in range(len(LIMBS))]
  norms = []
  #x_avg_lengths[0].append(1)
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

  limb_lengths_average = [sum(col)/len(col) for col in x_avg_lengths]
  norm_average = sum(norms)/len(norms)
  return limb_lengths_average, norm_average

def calculate_average_angles(x_coords):
  x_avg_angles =[[] for i in range(len(ANGLES))]
  for frame in x_coords:
    for i, angle in enumerate(ANGLES):
      if (np.all(frame[angle[0]] != 0) and 
         np.all(frame[angle[1]] != 0) and 
         np.all(frame[angle[2]] != 0)):
        
        a = calculate_angle(frame[angle[0]],frame[angle[1]],frame[angle[2]])
        x_avg_angles[i].append(a)
  angles_average = [sum(col)/len(col) for col in x_avg_angles]
  return angles_average

def calculate_limb_lengths(x_coords, average_LIMBS, average_norm):
  x_lengths = []
  for frame in x_coords:
    x_person = []
    if (np.any(frame[1] == 0)) or (np.any(frame[8] == 0)):
      norm = average_norm
    else:
      norm = np.linalg.norm(frame[1] - frame[8])

    #x_person.append(1)
    for i, limb in enumerate(LIMBS):
      if (np.any(frame[limb[0]] == 0)) or (np.any(frame[limb[1]] == 0)):
        l = average_LIMBS[i]
        x_person.append(l)
      else:
        l = np.linalg.norm(frame[limb[0]] - frame[limb[1]]) / norm
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
    for i, angle in enumerate(ANGLES):
      if (np.any(frame[angle[0]] == 0) or 
         np.any(frame[angle[1]] == 0) or 
         np.any(frame[angle[2]] == 0)):
        a = average_ang[i]
        x_person.append(a)
      else:
        a = calculate_angle(frame[angle[0]],frame[angle[1]],frame[angle[2]])
        x_person.append(a)
    x_angles.append(x_person)
  return x_angles

def calculate_distance(x_coords_1, x_coords_2, average_norm, distance_type = "2P"):
    x_distance = []

    for frame1, frame2 in zip(x_coords_1,x_coords_2):
        if (np.any(frame1[1] == 0)) or (np.any(frame1[8] == 0)):
            norm = average_norm
        else:
            norm = np.linalg.norm(frame1[1] - frame1[8])
        if (distance_type == "2P"):
            if (np.any(frame1[1] == 0)) or (np.any(frame2[1] == 0)):
                d = np.linalg.norm(frame1[0] - frame2[0]) / norm
                x_distance.append(d)
            else:
                d = np.linalg.norm(frame1[1] - frame2[1]) / norm
                x_distance.append(d)
        elif (distance_type == "25P"):
            x_person = []
            moving_avg_d = 1
            for i in range(25):
                if (np.any(frame1[i] == 0) or np.any(frame2[i] == 0)):
                    moving_avg_d = (sum(x_person) + moving_avg_d) / (i + 1) 
                    d = moving_avg_d
                else: 
                    d = np.linalg.norm(frame1[i] - frame2[i]) / norm
                x_person.append(d)
            x_distance.append(x_person)
        elif (distance_type == "cross_dist"):
            x_person = []
            moving_avg_d = 1
            for i in range(15):
                if (np.any(frame1[i] == 0) or np.any(frame2[1] == 0)):
                    moving_avg_d = (sum(x_person) + moving_avg_d) / (i + 1) 
                    d = moving_avg_d
                else: 
                    d = np.linalg.norm(frame1[i] - frame2[1]) / norm
                x_person.append(d)
            x_distance.append(x_person)
        elif (distance_type == "ffa"):
            x_person = [[0 for x in range(15)] for y in range(15)] 
            moving_avg_d = 1
            for i in range(15):
                for j in range(15):
                    if (np.any(frame1[i] == 0) or np.any(frame2[j] == 0)):
                        moving_avg_d = (sum(x_person[i]) + moving_avg_d) / (j + 1) 
                        d = moving_avg_d
                    else: 
                        d = np.linalg.norm(frame1[i] - frame2[j]) / norm
                    x_person[i][j] = d
            x_distance.append(x_person)
    return x_distance

def polar_coords(x_coords_1, x_coords_2, average_norm):
    results = []
    for frame1, frame2 in zip(x_coords_1,x_coords_2):
        if (np.any(frame1[1] == 0)) or (np.any(frame1[8] == 0)):
            norm = average_norm
        else:
            norm = np.linalg.norm(frame1[1] - frame1[8])

        if (np.any(frame1[1] == 0)) or (np.any(frame2[1] == 0)):
            norm_vector = (1,0)
        else:
            norm_vector = frame1[1] - frame2[1]
        
        x_person_dist = [[0 for x in range(15)] for y in range(15)] 
        x_person_angle = [[0 for x in range(15)] for y in range(15)] 
        moving_avg_d = 1
        for i in range(15):
            for j in range(15):
                if (np.any(frame1[i] == 0) or np.any(frame2[j] == 0)):
                    moving_avg_d = (sum(x_person_dist[i]) + moving_avg_d) / (j + 1) 
                    d = moving_avg_d
                    a = 1
                else:
                    moving_avg_d = (sum(x_person_dist[i]) + moving_avg_d) / (j + 1)  
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
    for frame1, frame2 in zip(x_coords_1,x_coords_2):
        if (np.any(frame1[1] == 0)) or (np.any(frame1[8] == 0)):
            norm = average_norm
            norm_vector = (1,0)
            middle_point = (0.5, 0.5)
        else:
            norm = np.linalg.norm(frame1[1] - frame1[8])
            norm_vector = frame1[1] - frame2[1]
            middle_point = (frame1[1] + frame2[1]) / 2
        
        x_person_dist = [0 for y in range(15)] 
        x_person_angle = [0  for y in range(15)] 
        moving_avg_d = 1
        for i in range(15):
            if (np.any(frame2[i] == 0)):
                moving_avg_d = (sum(x_person_dist) + moving_avg_d) / (i + 1)  
                d = moving_avg_d
                a = 1
            else:
                moving_avg_d = (sum(x_person_dist) + moving_avg_d) / (i + 1)  
                d = np.linalg.norm(middle_point - frame2[i]) / norm
                vector = middle_point - frame2[i]
                a = angle_between_vectors(norm_vector, vector)
            x_person_dist[i] = d
            x_person_angle[i] = a

        result = np.array(x_person_dist + x_person_angle).flatten()
        results.append(result)

    return results


def normalize(x, feature_type = FEATURES_TYPE_1, architecture_type = SINGLE_LSTM):
    #maybe add minmax scaling
    #removing c values
    av_angles = read_txt_to_list('/media/sebastian/STORAGE_HDD/data/average_angles.txt')
    av_norm = read_txt_to_list('/media/sebastian/STORAGE_HDD/data/average_norm.txt')
    av_limbs = read_txt_to_list('/media/sebastian/STORAGE_HDD/data/average_limbs.txt')
    print(x)
    x  = x.drop(x.columns[2::3], axis=1)
    #spliting data
    x_per1 = x.drop(x.columns[50:], axis=1)
    x_per2 = x.drop(x.columns[:50], axis=1)

    x_coords_per1 = []
    for x in x_per1.to_numpy():
        it = iter(x)
        joint_coords = list(zip(it,it))
        x_coords_per1.append(joint_coords)

    x_coords_per1 = np.array(x_coords_per1)

    x_coords_per2 = []
    for x in x_per2.to_numpy():
        it = iter(x)
        joint_coords = list(zip(it,it))
        x_coords_per2.append(joint_coords)

    x_coords_per2 = np.array(x_coords_per2)

    #Calculating new features
    if feature_type != FEATURES_TYPE_2 and feature_type != FEATURES_TYPE_3:
        x_angles_per1 = calculate_angles(x_coords_per1, av_angles)
        x_angles_per2 = calculate_angles(x_coords_per2, av_angles)
        x_limbs_per1 = calculate_limb_lengths(x_coords_per1,av_limbs, av_norm)
        x_limbs_per2 = calculate_limb_lengths(x_coords_per2,av_limbs, av_norm)

    # Feature type 1 - angles and lengths for both persons concatted into 1 big df
    if feature_type == FEATURES_TYPE_1 and architecture_type == SINGLE_LSTM:
        x_distances = calculate_distance(x_coords_per1, x_coords_per2, av_norm, distance_type = "25P")
        x_angles_per1_df = pd.DataFrame(x_angles_per1)
        x_angles_per2_df = pd.DataFrame(x_angles_per2)
        x_limbs_per1_df = pd.DataFrame(x_limbs_per1)
        x_limbs_per2_df = pd.DataFrame(x_limbs_per2)
        x_distances_df = pd.DataFrame(x_distances)
        x_features_typeP = pd.concat([x_angles_per1_df, x_angles_per2_df, x_limbs_per1_df, x_limbs_per2_df, x_distances_df], axis=1)
        x_features_typeP.columns = range(0,x_features_typeP.shape[1])

        return x_features_typeP

    # Feature type 2 - angles and lengths are separate for each person and 
    # distance is also separate
    elif feature_type == FEATURES_TYPE_2 and architecture_type == TRIPLE_LSTM:

        x_distances = calculate_distance(x_coords_per1, x_coords_per2, av_norm, distance_type = "25P")
        x_angles_per1_df = pd.DataFrame(x_angles_per1)
        x_angles_per2_df = pd.DataFrame(x_angles_per2)
        x_limbs_per1_df = pd.DataFrame(x_limbs_per1)
        x_limbs_per2_df = pd.DataFrame(x_limbs_per2)
        x_distances_df = pd.DataFrame(x_distances)


        x_features_typeV_per1 = pd.concat([x_angles_per1_df, x_limbs_per1_df], axis=1)
        x_features_typeV_per1.columns = range(0,x_features_typeV_per1.shape[1])
        x_features_typeV_per2 = pd.concat([x_angles_per2_df, x_limbs_per2_df], axis=1)
        x_features_typeV_per2.columns = range(0,x_features_typeV_per2.shape[1])
        x_distances_df.columns = range(0,x_distances_df.shape[1])

        return (x_features_typeV_per1, x_features_typeV_per2, x_distances_df)


    # Feature type 3 - angles, lengths and distances are separate for each person 
    elif feature_type == FEATURES_TYPE_3 and architecture_type == DOUBLE_LSTM:
        x_distance_per1 = calculate_distance(x_coords_per1, x_coords_per2, av_norm, distance_type = "cross_dist")
        x_distance_per2 = calculate_distance(x_coords_per2, x_coords_per1, av_norm, distance_type = "cross_dist")
        x_angles_per1_df = pd.DataFrame(x_angles_per1)
        x_angles_per2_df = pd.DataFrame(x_angles_per2)
        x_limbs_per1_df = pd.DataFrame(x_limbs_per1)
        x_limbs_per2_df = pd.DataFrame(x_limbs_per2)
        x_distance_per1_df = pd.DataFrame(x_distance_per1)
        x_distance_per2_df = pd.DataFrame(x_distance_per2)

        x_features_typeD_per1 = pd.concat([x_angles_per1_df, x_limbs_per1_df, x_distance_per1_df], axis=1)
        x_features_typeD_per1.columns = range(0,x_features_typeD_per1.shape[1])
        x_features_typeD_per2 = pd.concat([x_angles_per2_df, x_limbs_per2_df, x_distance_per2_df], axis=1)
        x_features_typeD_per2.columns = range(0,x_features_typeD_per2.shape[1])

        return (x_features_typeD_per1, x_features_typeD_per2)

    elif feature_type == FEATURES_TYPE_2 and architecture_type == SINGLE_LSTM:

        x_polar_coords1 = polar_coords(x_coords_per1, x_coords_per2, av_norm)
        x_polar_coords2 = polar_coords(x_coords_per2, x_coords_per1, av_norm)
        x_polar_coords1_df = pd.DataFrame(x_polar_coords1)
        x_polar_coords2_df = pd.DataFrame(x_polar_coords2)
        x = pd.concat([x_polar_coords1_df, x_polar_coords2_df], axis=1)
        x.columns = range(0,x.shape[1])
        return x

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
        x = pd.concat([x_polar_coords1_df, x_polar_coords2_df], axis=1)
        x.columns = range(0,x.shape[1])
        return x

    elif feature_type == FEATURES_TYPE_3 and architecture_type == TRIPLE_LSTM:
        x_polar_coords1 = polar_coords2(x_coords_per1, x_coords_per2, av_norm)
        x_polar_coords2 = polar_coords2(x_coords_per2, x_coords_per1, av_norm)
        x_polar_coords1_df = pd.DataFrame(x_polar_coords1)
        x_polar_coords2_df = pd.DataFrame(x_polar_coords2)
        return (x_polar_coords1_df, x_polar_coords2_df)


def read_txt_to_list(path: str) -> List[Type[Any]]:
    out = []
    with open(path, 'r') as f:
        for line in f.readlines():
            out.append(line.strip("\n"))
    return out




