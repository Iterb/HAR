import sys

import numpy as np

sys.path.append(".")
from engine.deep_sort import nn_matching, preprocessing
from engine.deep_sort.detection import Detection
from engine.deep_sort.tracker import Tracker as DeepTracker
from utils import generate_detections as gdet


class Tracker:
    def __init__(self, cfg, width, height):
        model_filename = "engine/deep_sort/model_data/mars-small128.pb"
        self.encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        max_cosine_distance = cfg.TRACKER.MAX_COSINE_DISTANCE
        nn_budget = cfg.TRACKER.NN_BUDGET
        self.nms_max_overlap = cfg.TRACKER.NMS_MAX_OVERLAP
        max_age = cfg.TRACKER.MAX_AGE
        n_init = cfg.TRACKER.N_INIT
        self.width = width
        self.height = height
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget
        )
        self.tracker = DeepTracker(metric, max_age=max_age, n_init=n_init)

    def create_bboxes(self, poses):
        global seen_bodyparts
        """
        Parameters
        ----------
        poses: ndarray of human 2D poses [People * BodyPart]
        Returns
        ----------
        boxes: ndarray of containing boxes [People * [x1,y1,x2,y2]]
        """
        boxes = []
        for person in poses:
            seen_bodyparts = person[np.where((person[:, 0] != 0) | (person[:, 1] != 0))]
            mean = np.mean(seen_bodyparts, axis=0)
            deviation = 2 * np.std(seen_bodyparts, axis=0)
            box = [
                int(mean[0] - deviation[0]),
                int(mean[1] - deviation[1]),
                int(mean[0] + deviation[0]),
                int(mean[1] + deviation[1]),
            ]
            boxes.append(box)
        return np.array(boxes)

    def track(self, datum):
        keypoints, currentFrame = np.array(datum.poseKeypoints), datum.cvOutputData
        poses = self.denormalize_poses(keypoints[:, :, :2])
        boxes = self.create_bboxes(poses)
        boxes_xywh = [[x1, y1, x2 - x1, y2 - y1] for [x1, y1, x2, y2] in boxes]
        features = self.encoder(currentFrame, boxes_xywh)
        nonempty = lambda xywh: xywh[2] != 0 and xywh[3] != 0
        detections = [
            Detection(bbox, 1.0, feature, pose)
            for bbox, feature, pose in zip(boxes_xywh, features, poses)
            if nonempty(bbox)
        ]
        # Run non-maxima suppression.
        boxes_det = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes_det, self.nms_max_overlap, scores
        )
        detections = [detections[i] for i in indices]
        # Call the tracker
        self.tracker.predict()
        self.tracker.update(currentFrame, detections)

        return self.tracker.tracks, currentFrame

    def denormalize_poses(self, poses):
        return poses * np.array([self.width, self.height])
