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
    print('GPU=', GPU)


def get_dm():
    # set env variable for data
    DIR_INPUT = cfg["data_path"]
    os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT
    dm = LocalDataManager(None)
    return dm


def get_device():
    gpu = None
    if not ('GPU' in globals()):
        gpu = True
    else:
        gpu = GPU

    print('GPU =', gpu)

    if gpu is False:
        return "cpu"
    else:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_val_data():
    dm = get_dm()

    eval_cfg = cfg["val_data_loader"]

    # MIN_FUTURE_STEPS = 10
    # num_frames_to_chop = cfg['model_params']['history_num_frames']+1

    # eval_base_path = create_chopped_dataset(dm.require(eval_cfg["key"]),
    #                                         cfg["raster_params"]["filter_agents_threshold"],
    #                                         num_frames_to_chop,
    #                                         cfg["model_params"]["future_num_frames"],
    #                                         MIN_FUTURE_STEPS)

    eval_base_path = '/home/axot/lyft/data/scenes/validate_chopped_31'

    eval_zarr_path = str(Path(eval_base_path) /
                         Path(dm.require(eval_cfg["key"])).name)
    eval_mask_path = str(Path(eval_base_path) / "mask.npz")
    eval_gt_path = str(Path(eval_base_path) / "gt.csv")

    rasterizer = build_rasterizer(cfg, dm)
    eval_zarr = ChunkedDataset(eval_zarr_path).open()
    eval_mask = np.load(eval_mask_path)["arr_0"]
    # ===== INIT DATASET AND LOAD MASK
    eval_dataset = AgentDataset(cfg, eval_zarr, rasterizer, agents_mask=eval_mask)
    # eval_dataset = AgentDataset(cfg, eval_zarr, rasterizer)
    eval_dataloader = DataLoader(eval_dataset,
                                 shuffle=eval_cfg["shuffle"],
                                 batch_size=eval_cfg["batch_size"],
                                 num_workers=eval_cfg["num_workers"])

    return eval_dataloader


def get_model(cfg):
    model = LyftModel(cfg)
    # load weight if there is a pretrained model
    if cfg["model_params"]["weight_path"]:
        load_pretrained(model, cfg)
        print('load_pretrained from:', cfg["model_params"]["weight_path"])

    # print('model =', model)

    return model


def validation(model, device):

    model.eval()
    torch.set_grad_enabled(False)

    # store information for evaluation
    future_coords_offsets_pd = []
    timestamps = []
    agent_ids = []
    confidences_list = []

    val_dataloader = load_val_data()
    # num_iter = iter(val_dataloader)
    # progress_bar = tqdm(range(cfg["train_params"]["val_num_steps"]))
    progress_bar = tqdm(val_dataloader)

    for data in progress_bar:
        # data = next(num_iter)
        preds, confs = forward(data, model, device)

        # convert agent coordinates into world offsets
        preds = preds.cpu().numpy()

        confs = confs.cpu().numpy()
        world_from_agents = data["world_from_agent"].numpy()
        centroids = data["centroid"].numpy()
        coords_offset = []

        for pred, world_from_agent, centroid in zip(preds, world_from_agents, centroids):
            for mode in range(3):
                pred[mode] = transform_points(
                    pred[mode], world_from_agent) - centroid[:2]
            coords_offset.append(pred)

        confidences_list.append(confs)
        future_coords_offsets_pd.append(np.stack(coords_offset))
        timestamps.append(data["timestamp"].numpy().copy())
        agent_ids.append(data["track_id"].numpy().copy())

    pred_path = f"{gettempdir()}/pred.csv"
    write_pred_csv(pred_path,
                   timestamps=np.concatenate(timestamps),
                   track_ids=np.concatenate(agent_ids),
                   coords=np.concatenate(future_coords_offsets_pd),
                   confs=np.concatenate(confidences_list)
                   )

    eval_base_path = '/home/axot/lyft/data/scenes/validate_chopped_31'
    eval_gt_path = str(Path(eval_base_path) / "gt.csv")

    metrics = compute_metrics_csv(eval_gt_path, pred_path, [
                                  neg_multi_log_likelihood])
    for metric_name, metric_mean in metrics.items():
        print(metric_name, metric_mean)
    return metrics['neg_multi_log_likelihood']


def main():
    args = parse_args()
    initialize(args.exp_id)
    model = get_model(cfg)
    model.to(get_device())
    validation(model, get_device())


if __name__ == "__main__":
    main()
