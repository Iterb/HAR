from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()

_C.MODEL = CN()
_C.MODEL.DROPOUT_RATE = 0.2
_C.MODEL.ARCH = "double" # single/double/triple
_C.MODEL.NAME = "double_lstm_F3_full" 

#_C.MODEL.CLASSES = ({'punch':0, 'kicking':1, 'pushing': 2, 'pat on back' : 3, 'point finger' : 4, 'hugging' : 5, 'giving an object' : 6, 'touch pocket' : 7, 'shaking hands' : 8, 'walking towards' : 9, 'walking apart' :10})
# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C.SEQUENCE = CN()
_C.SEQUENCE.WINDOW_SIZE = 45
_C.SEQUENCE.LIN_SIZE = 55
_C.SEQUENCE.LABEL_OFFSET = 0
# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
_C.DATASETS.FEATURES_TYPE = 3
# CSV of all key points in all videos from NTU dataset
_C.DATASETS.FULL = ("/media/sebastian/STORAGE_HDD/data/rose_data_pc_2.csv") #
_C.DATASETS.FEATURES_FULL = ("/media/sebastian/STORAGE_HDD/data/normalized_25P_short.csv")
_C.DATASETS.FEATURES2_FULL = ("/home/sebastian/projects/HAR/polarcoords_pera.csv")
_C.DATASETS.FEATURES3_FULL = ("/media/sebastian/STORAGE_HDD/data/polarcoords2_full.csv")
# features for double LSTM
_C.DATASETS.FEATURES_PER1_D = ("/media/sebastian/STORAGE_HDD/data/normalized_25P_short.csv")
_C.DATASETS.FEATURES_PER2_D = ("/media/sebastian/STORAGE_HDD/data/normalized_25P_short.csv")
_C.DATASETS.FEATURES2_PER1_D = ("/media/sebastian/STORAGE_HDD/data/polarcoords_pera.csv")
_C.DATASETS.FEATURES2_PER2_D = ("/media/sebastian/STORAGE_HDD/data/polarcoords_perb.csv")
_C.DATASETS.FEATURES3_PER1_D = ("/media/sebastian/STORAGE_HDD/data/polarcoords2_pera.csv")
_C.DATASETS.FEATURES3_PER2_D = ("/media/sebastian/STORAGE_HDD/data/polarcoords2_perb.csv")
# features for triple LSTM
_C.DATASETS.FEATURES_PER1 = ("/media/sebastian/STORAGE_HDD/data/normalized_25P_short.csv")
_C.DATASETS.FEATURES_PER2 = ("/media/sebastian/STORAGE_HDD/data/normalized_25P_short.csv")
_C.DATASETS.FEATURES_DIST = ("/media/sebastian/STORAGE_HDD/data/normalized_25P_short.csv")
# List of the dataset names for testing
_C.DATASETS.TEST = ("data/datasets/test.csv")
_C.DATASETS.SPLIT_TYPE = 'cv' # cv or cs
_C.DATASETS.TRAIN_SUBJECTS = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
_C.DATASETS.TRAIN_CAMERAS = [2,3] 

# ---------------------------------------------------------------------------- #
# Inference
# ---------------------------------------------------------------------------- #
_C.INFER = CN()
_C.INFER.BATCH_SIZE = 1
_C.INFER.FEATURES_TYPE = 3
_C.INFER.ARCH = "single" 
_C.INFER.WINDOW_SIZE = 15
_C.INFER.VIDEO_PATH = ('/media/sebastian/STORAGE_HDD/data/tv_human_interactions_videos/handShake_0020.avi')


# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER_NAME = "Adam"
_C.SOLVER.MAX_EPOCHS = 55
_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.BATCH_SIZE = 64

_C.TEST = CN()
_C.TEST.BATCH_SIZE = 1
_C.TEST.WEIGHT = ""

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "output"
