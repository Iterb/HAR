from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()

_C.MODEL = CN()
_C.MODEL.DROPOUT_RATE = 0.3
_C.MODEL.ARCH = "convnet"  # single/double/triple/convnet
_C.MODEL.LSTM_LAYERS = 3  # 1/2/3

# _C.MODEL.CLASSES = ({'punch':0, 'kicking':1, 'pushing': 2, 'pat on back' : 3, 'point finger' : 4, 'hugging' : 5, 'giving an object' : 6, 'touch pocket' : 7, 'shaking hands' : 8, 'walking towards' : 9, 'walking apart' :10})
# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C.SEQUENCE = CN()
_C.SEQUENCE.WINDOW_SIZE = 25
_C.SEQUENCE.LIN_SIZE = 25
_C.SEQUENCE.LABEL_OFFSET = 0
# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
_C.DATASETS.FEATURES_TYPE = 2
# CSV of all key points in all videos from NTU dataset
_C.DATASETS.FULL = "/media/sebastian/STORAGE_HDD/data/rose_data_pc_2.csv"  #
_C.DATASETS.FEATURES_FULL = "/media/sebastian/STORAGE_HDD/data/normalized_25P_short.csv"
_C.DATASETS.FEATURES2_FULL = "polarcoords_pera.csv"
_C.DATASETS.FEATURES3_FULL = "/media/sebastian/STORAGE_HDD/data/polarcoords2_full.csv"
# features for double LSTM
_C.DATASETS.FEATURES_PER1_D = (
    "/media/sebastian/STORAGE_HDD/data/normalized_25P_short.csv"
)
_C.DATASETS.FEATURES_PER2_D = (
    "/media/sebastian/STORAGE_HDD/data/normalized_25P_short.csv"
)
_C.DATASETS.FEATURES2_PER1_D = "/media/sebastian/STORAGE_HDD/data/polarcoords_pera.csv"
_C.DATASETS.FEATURES2_PER2_D = "/media/sebastian/STORAGE_HDD/data/polarcoords_perb.csv"
_C.DATASETS.FEATURES3_PER1_D = "/media/sebastian/STORAGE_HDD/data/polarcoords2_pera.csv"
_C.DATASETS.FEATURES3_PER2_D = "/media/sebastian/STORAGE_HDD/data/polarcoords2_perb.csv"
# features for triple LSTM
_C.DATASETS.FEATURES_PER1 = "/media/sebastian/STORAGE_HDD/data/normalized_25P_short.csv"
_C.DATASETS.FEATURES_PER2 = "/media/sebastian/STORAGE_HDD/data/normalized_25P_short.csv"
_C.DATASETS.FEATURES_DIST = "/media/sebastian/STORAGE_HDD/data/normalized_25P_short.csv"
# List of the dataset names for testing
_C.DATASETS.TEST = "data/datasets/test.csv"
_C.DATASETS.SPLIT_TYPE = "cv"  # cv or cs
_C.DATASETS.TRAIN_SUBJECTS = [
    1,
    2,
    4,
    5,
    8,
    9,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    25,
    27,
    28,
    31,
    34,
    35,
    38,
]
_C.DATASETS.TRAIN_CAMERAS = [2, 3]

# ---------------------------------------------------------------------------- #
# Inference
# ---------------------------------------------------------------------------- #
_C.INFER = CN()
_C.INFER.BATCH_SIZE = 1
_C.INFER.FEATURES_TYPE = 3
_C.INFER.ARCH = "single"
_C.INFER.WINDOW_SIZE = 40
_C.INFER.VIDEO_PATH = (
    "/media/sebastian/STORAGE_HDD/data/human_interaction/cropped_many_actions3.avi"
)
_C.INFER.MODEL_PATH = "/workspace/models/double_lstm_F3_full_061120212131.onnx"
_C.INFER.DO_TRACK = True
_C.INFER.DO_SHOW = True
_C.INFER.WINDOW_DURATION_S = 3
_C.INFER.WINDOW_OFFSET_S = 1
_C.INFER.OUTPUT_PATH = "/workspace/output.avi"
# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER_NAME = "Adam"
_C.SOLVER.MAX_EPOCHS = 55
_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.BATCH_SIZE = 256

_C.TEST = CN()
_C.TEST.BATCH_SIZE = 1
_C.TEST.WEIGHT = ""

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "output"

# ---------------------------------------------------------------------------- #
# Openpose options
# ---------------------------------------------------------------------------- #
_C.OPENPOSE = CN()
_C.OPENPOSE.MODELS_PATH = "/home/marcelo/openpose/models/"

# ---------------------------------------------------------------------------- #
# Tracker options
# ---------------------------------------------------------------------------- #
_C.TRACKER = CN()
_C.TRACKER.MAX_COSINE_DISTANCE = 1
_C.TRACKER.NN_BUDGET = None
_C.TRACKER.NMS_MAX_OVERLAP = 1.0
_C.TRACKER.MAX_AGE = 100
_C.TRACKER.N_INIT = 20

_C.MODEL.NAME = f"{_C.MODEL.ARCH}_F{_C.DATASETS.FEATURES_TYPE }_full"
