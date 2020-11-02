import os
import random
import time
import argparse
from tqdm import tqdm
from tempfile import gettempdir

import numpy as np
import pandas as pd
import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
from ranger import Ranger  # this is from ranger.py
from ranger import RangerVA  # this is from ranger913A.py
from ranger import RangerQH  # this is from rangerqh.py


from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset
from l5kit.rasterization import build_rasterizer
from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, create_chopped_dataset
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace
from l5kit.geometry import transform_points
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp_id', type=str, default='', required=True,
                        help='path of your experiment directory name')
    return parser.parse_args()


def initialize(exp_id):

    # seed = int(time.time())
    seed = int(time.time())
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    np.set_printoptions(precision=4)
    # load experiment config and model architecture
    module_path = f'experiment.{exp_id}'
    exec(f'from {module_path}.config import *', globals())
    exec(f'from {module_path}.model import LyftModel', globals())
    exec(f'from {module_path}.model import forward', globals())
    try:
        exec(f'from {module_path}.model import load_pretrained', globals())
    except Exception as e:
        print(f"There is no load_pretrained function in {module_path}.")
    print('GPU =', GPU)


def get_dm():
    # set env variable for data
    DIR_INPUT = cfg["data_path"]
    os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT
    dm = LocalDataManager(None)
    return dm


def load_sample_data():
    # load training data
    dm = get_dm()
    sample_cfg = cfg["sample_data_loader"]

    rasterizer = build_rasterizer(cfg, dm)
    sample_zarr = ChunkedDataset(dm.require(sample_cfg["key"])).open()
    sample_dataset = AgentDataset(cfg, sample_zarr, rasterizer)
    sample_dataloader = DataLoader(sample_dataset,
                                   shuffle=sample_cfg["shuffle"],
                                   batch_size=sample_cfg["batch_size"],
                                   num_workers=sample_cfg["num_workers"])
    print('len(sample_data_loader):', len(sample_dataloader))
    return sample_dataloader


def build_chopped_dataset():
    dm = get_dm()

    eval_cfg = cfg["val_data_loader"]

    MIN_FUTURE_STEPS = 10
    num_frames_to_chop = [80, 130, 180]
    for chop_frame in num_frames_to_chop:
        eval_base_path = create_chopped_dataset(dm.require(eval_cfg["key"]),
                                                cfg["raster_params"]["filter_agents_threshold"],
                                                chop_frame,
                                                cfg["model_params"]["future_num_frames"],
                                                MIN_FUTURE_STEPS)
        print(eval_base_path)


def main():
    args = parse_args()
    initialize(args.exp_id)
    build_chopped_dataset()


if __name__ == "__main__":
    main()
