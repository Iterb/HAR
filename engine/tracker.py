import sys
import cv2
import numpy as np

sys.path.append('.')
from engine.deep_sort.detection import Detection
from engine.deep_sort.tracker import Tracker as DeepTracker
from engine.deep_sort import nn_matching
from engine.deep_sort import preprocessing
from utils.utils import poses2boxes
from utils import generate_detections as gdet

class Tracker():
    def __init__(self, cfg, width, height):
        model_filename = 'engine/deep_sort/model_data/mars-small128.pb'
        self.encoder = gdet.create_box_encoder(model_filename,batch_size=1)
        max_cosine_distance = cfg.TRACKER.MAX_COSINE_DISTANCE
        nn_budget = cfg.TRACKER.NN_BUDGET
        self.nms_max_overlap = cfg.TRACKER.NMS_MAX_OVERLAP
        max_age = cfg.TRACKER.MAX_AGE
        n_init = cfg.TRACKER.N_INIT
        self.width = width
        self.height = height
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget)
        self.tracker = DeepTracker(metric, max_age = max_age,n_init= n_init)

    def run(self, datum):
        keypoints, currentFrame = np.array(datum.poseKeypoints), datum.cvOutputData
        poses = self.denormalize_poses(keypoints[:,:,:2])
        boxes = poses2boxes(poses)
        boxes_xywh = [[x1,y1,x2-x1,y2-y1] for [x1,y1,x2,y2] in boxes]
        features = self.encoder(currentFrame, boxes_xywh)
        nonempty = lambda xywh: xywh[2] != 0 and xywh[3] != 0
        detections = [Detection(bbox, 1.0, feature, pose) for bbox, feature, pose in zip(boxes_xywh, features, poses) if nonempty(bbox)]
        # Run non-maxima suppression.
        boxes_det = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes_det, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        # Call the tracker
        self.tracker.predict()
        self.tracker.update(currentFrame, detections)

        return self.tracker.tracks, currentFrame

    def denormalize_poses(self, poses):
        unposes = poses * np.array([self.width, self.height])
        return unposes
