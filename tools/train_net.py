# encoding: utf-8

import argparse
import logging
import os
import sys
from os import mkdir

sys.path.append(".")
from config import cfg
from data.dataset import Dataset
from engine import Summarizer, Tester, Trainer
from modeling import DoubleLSTM, SingleLSTM, TripleLSTM
from utils.logger import setup_logger
from utils.wandblog import setup_wandb_logger


def do_train(cfg):

    run = setup_wandb_logger(cfg)
    dataset = Dataset(cfg)

    if cfg.MODEL.ARCH == "single":
        model = SingleLSTM(cfg).compile_model()
    elif cfg.MODEL.ARCH == "double":
        model = DoubleLSTM(cfg).compile_model()
    elif cfg.MODEL.ARCH == "triple":
        model = TripleLSTM(cfg).compile_model()

    trainer = Trainer(model, dataset, cfg)
    trained_model = trainer.fit()
    accuracy = Tester.evaluate(trained_model, dataset, cfg)
    summarizer = Summarizer(trained_model, dataset, cfg)
    classification_report = summarizer.classification_report()
    logger = logging.getLogger("model.train")
    logger.info("{}".format(classification_report))
    cm = summarizer.create_cm()
    logger.info("{}".format(cm))

    onnx_model = trainer.onnx_export()


def main():
    parser = argparse.ArgumentParser(description="Keras training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = setup_logger("model", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)
    logger.propagate = False

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, "r") as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    do_train(cfg)


if __name__ == "__main__":
    main()
