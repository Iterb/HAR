# encoding: utf-8

import argparse
import os
import sys
from os import mkdir

sys.path.append(".")
sys.path.append("/openpose/examples/openpose-examples")
from config import cfg
from engine import Inference, VideoHander
from utils.logger import setup_logger


def arg_parser():
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

    return parser.parse_args()


def do_inference(cfg):
    inference_session = Inference(cfg, cfg.INFER.PATH_TO_ONNX)
    prediction = inference_session.predict()
    VideoHander.save_video(cfg, prediction)


def main():
    args = arg_parser()
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
        with open(args.config_file, "r") as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)

    do_inference(cfg)


if __name__ == "__main__":
    main()
