import wandb


def setup_wandb_logger(cfg):

    wandb.login()
    run = wandb.init(project='human_activity_recognition',
           config={
              "learning_rate": cfg.SOLVER.BASE_LR,
              "epochs": cfg.SOLVER.MAX_EPOCHS,
              "batch_size": cfg.SOLVER.BATCH_SIZE ,
              "architecture": cfg.MODEL.ARCH,
              "features_type":cfg.DATASETS.FEATURES_TYPE,
              "number_of_frames": cfg.SEQUENCE.LIN_SIZE,
              "Test_train_split" : cfg.DATASETS.SPLIT_TYPE,
              "Dropout": cfg.MODEL.DROPOUT_RATE
           })
    return run