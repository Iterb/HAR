from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()

_C.MODEL = CN()
_C.MODEL.ARCH = "single" # single/double/triple
_C.MODEL.NAME = "single_lstm_F1" 
_C.MODEL.FEATURES_TYPE = '1'
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
# CSV of all key points in all videos from NTU dataset
_C.DATASETS.FULL = ("/media/sebastian/STORAGE_HDD/data/rose_data_pc_2.csv") #
_C.DATASETS.FEATURES_FULL = ("/media/sebastian/STORAGE_HDD/data/normalized_25P_short.csv")
# features for double LSTM
_C.DATASETS.FEATURES2_PER1 = ("/media/sebastian/STORAGE_HDD/data/normalized_25P_short.csv")
_C.DATASETS.FEATURES2_PER2 = ("/media/sebastian/STORAGE_HDD/data/normalized_25P_short.csv")
# List of the dataset names for testing
_C.DATASETS.TEST = ("data/datasets/test.csv")
_C.DATASETS.SPLIT_TYPE = 'cs' # cv or cs
_C.DATASETS.TRAIN_SUBJECTS = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
_C.DATASETS.TRAIN_CAMERAS = [2,3] 
# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER_NAME = "Adam"
_C.SOLVER.MAX_EPOCHS = 20
_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.BATCH_SIZE = 64

_C.TEST = CN()
_C.TEST.BATCH_SIZE = 1
_C.TEST.WEIGHT = ""


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "output"
