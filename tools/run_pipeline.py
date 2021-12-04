# encoding: utf-8

import argparse
import os
import sys
import subprocess
from os import mkdir
import numpy as np

sys.path.append(".")
sys.path.append("/openpose/examples/openpose-examples")
from config import cfg
from utils.logger import setup_logger
from engine.estimate_pose import (
    extract_pose_features_from_video,
    calculate_stacked_preditions,
)
from engine.inference import do_inference
from utils.vizualize import plot_probablites
from utils.display import put_interactions_on_video


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

    args = parser.parse_args()

    return args


def main():
    args = arg_parser()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    # if args.config_file != "":
    #     cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)
    # cfg.freeze()

    # output_dir = cfg.OUTPUT_DIR
    # if output_dir and not os.path.exists(output_dir):
    #     mkdir(output_dir)

    # logger = setup_logger("model", output_dir, 0)
    # logger.info("Using {} GPUS".format(num_gpus))
    # logger.info(args)
    # logger.propagate = False

    # if args.config_file != "":
    #     with open(args.config_file, 'r') as cf:
    #         config_str = "\n" + cf.read()
    #         logger.info(config_str)

    pose_features = extract_pose_features_from_video(cfg, save_output=True)
    preds = []
    for keypoints in pose_features:
        preds.append(do_inference(cfg, keypoints))
    stacked_preds = calculate_stacked_preditions(
        cfg.INFER.WINDOW_DURATION_S, cfg.INFER.WINDOW_OFFSET_S, preds
    )
    pose_features = extract_pose_features_from_video(
        cfg, save_output=True, full_video=True
    )
    print(stacked_preds)
    print(stacked_preds.shape)
    plot_probablites(stacked_preds)
    print(np.sum(stacked_preds, axis=1))
    put_interactions_on_video(
        cfg.INFER.OUTPUT_PATH,
        stacked_preds,
        cfg.INFER.WINDOW_DURATION_S,
        cfg.INFER.WINDOW_OFFSET_S,
        cfg.INFER.OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
