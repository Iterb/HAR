import wandb


def setup_wandb_logger(cfg):

    wandb.login()
    return wandb.init(
        project="human_activity_recognition",
        config={
            "learning_rate": cfg.SOLVER.BASE_LR,
            "epochs": cfg.SOLVER.MAX_EPOCHS,
            "batch_size": cfg.SOLVER.BATCH_SIZE,
            "architecture": cfg.MODEL.ARCH,
            "name": cfg.MODEL.NAME,
            "features_type": cfg.DATASETS.FEATURES_TYPE,
            "number_of_frames": cfg.SEQUENCE.LIN_SIZE,
            "Test_train_split": cfg.DATASETS.SPLIT_TYPE,
            "dropout": cfg.MODEL.DROPOUT_RATE,
            "num_of_lstm_layers": cfg.MODEL.LSTM_LAYERS,
            "LSTM_size": cfg.MODEL.LSTM_SIZE,
        },
    )
