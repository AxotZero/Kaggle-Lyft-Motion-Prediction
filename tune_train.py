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
from collections import OrderedDict
from l5kit.evaluation.csv_utils import read_gt_csv

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


def load_test_data():
    dm = get_dm()

    # print(eval_dataset)
    test_cfg = cfg["test_data_loader"]
    rasterizer = build_rasterizer(cfg, dm)
    test_zarr = ChunkedDataset(dm.require(test_cfg["key"])).open()
    test_mask = np.load(f'{cfg["data_path"]}/scenes/mask.npz')["arr_0"]
    test_dataset = AgentDataset(
        cfg, test_zarr, rasterizer, agents_mask=test_mask)
    test_dataloader = DataLoader(test_dataset,
                                 shuffle=test_cfg["shuffle"],
                                 batch_size=test_cfg["batch_size"],
                                 num_workers=test_cfg["num_workers"])
    return test_dataloader


def load_tune_data():
    dm = get_dm()

    eval_cfg = cfg["val_data_loader"]

    eval_base_path = '/home/axot/lyft/data/scenes/validate_chopped_31'

    eval_zarr_path = str(Path(eval_base_path) /
                         Path(dm.require(eval_cfg["key"])).name)
    eval_mask_path = str(Path(eval_base_path) / "mask.npz")
    eval_gt_path = str(Path(eval_base_path) / "gt.csv")

    rasterizer = build_rasterizer(cfg, dm)
    eval_zarr = ChunkedDataset(eval_zarr_path).open()
    eval_mask = np.load(eval_mask_path)["arr_0"]
    # ===== INIT DATASET AND LOAD MASK
    eval_dataset = AgentDataset(
        cfg, eval_zarr, rasterizer, agents_mask=eval_mask)

    gt_dict = OrderedDict()
    for el in read_gt_csv(eval_gt_path):
        gt_dict[el["track_id"] + el["timestamp"]] = el
    
    eval_dataloader = DataLoader(eval_dataset,
                                 shuffle=eval_cfg["shuffle"],
                                 batch_size=eval_cfg["batch_size"],
                                 num_workers=eval_cfg["num_workers"])

    return eval_dataloader, gt_dict


def get_model(cfg):
    model = LyftModel(cfg)
    # load weight if there is a pretrained model
    if cfg["model_params"]["weight_path"]:
        load_pretrained(model, cfg)
        print('load_pretrained from:', cfg["model_params"]["weight_path"])

    # print('model =', model)

    return model


def criterion(gt: Tensor, pred: Tensor, confidences: Tensor, avails: Tensor) -> Tensor:
    """
    Compute a negative log-likelihood for the multi-modal scenario.
    log-sum-exp trick is used here to avoid underflow and overflow, For more information about it see:
    https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
    https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    https://leimao.github.io/blog/LogSumExp/
    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        pred (Tensor): array of shape (bs)x(modes)x(time)x(2D coords)
        confidences (Tensor): array of shape (bs)x(modes) with a confidence for each mode in each sample
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """
    assert len(pred.shape) == 4, f"expected 3D (MxTxC) array for pred, got {pred.shape}"
    
    batch_size, num_modes, future_len, num_coords = pred.shape

    assert gt.shape == (batch_size, future_len,
                        num_coords), f"expected 2D (Time x Coords) array for gt, got {gt.shape}"
    assert confidences.shape == (
        batch_size, num_modes), f"expected 1D (Modes) array for gt, got {confidences.shape}"

    # assert torch.allclose(torch.sum(confidences, dim=1) ,
    #                         confidences.new_ones((batch_size,)) ), "confidences should sum to 1"
    if not (torch.allclose(torch.sum(confidences, dim=1), confidences.new_ones((batch_size,)))):
        print('confidences:', confidences)
        print('torch.sum:', torch.sum(confidences, dim=1))
        # print('confidences:', confidences.new_ones((batch_size,)))
        raise "confidences should sum to 1"

    assert avails.shape == (batch_size, future_len), f"expected 1D (Time) array for gt, got {avails.shape}"
    # assert all data are valid
    assert torch.isfinite(pred).all(), "invalid value found in pred"
    assert torch.isfinite(gt).all(), "invalid value found in gt"
    assert torch.isfinite(confidences).all(), "invalid value found in confidences"
    assert torch.isfinite(avails).all(), "invalid value found in avails"

    # convert to (batch_size, num_modes, future_len, num_coords)
    gt = torch.unsqueeze(gt, 1)  # add modes
    avails = avails[:, None, :, None]  # add modes and cords

    # error (batch_size, num_modes, future_len)
    # reduce coords and use availability
    error = torch.sum(((gt - pred) * avails) ** 2, dim=-1)

    # when confidence is 0 log goes to -inf, but we're fine with it
    with np.errstate(divide="ignore"):
        # error (batch_size, num_modes)
        error = torch.log(confidences) - 0.5 * \
            torch.sum(error, dim=-1)  # reduce time

    # error are negative at this point, so max() gives the minimum one
    max_value, _ = error.max(dim=1, keepdim=True)

    error = -torch.log(torch.sum(torch.exp(error - max_value),
                                 dim=-1, keepdim=True)) - max_value  # reduce modes

    return torch.mean(error)


def transform_ts_points(points, transf_matrix):
    """
        points: bs * 3 * 50 * 2
        transf: bs * 3 * 3 * 3
    """
    num_dims = len(transf_matrix[0]) - 1

    transf_matrix = torch.transpose(transf_matrix, 2, 3)

    ret = torch.matmul(points, transf_matrix[:, :, :num_dims, :num_dims])

    adding = transf_matrix[:, :, -1, :num_dims]
    adding = adding.unsqueeze(2) # bs * 1 * 3 * 3
    adding = adding.repeat(1, 1, 50, 1) # bs * 1 * 3 * 3

    ret = ret +  adding
    return  ret


def train(model, exp_id):
    save_model_dir = f'experiment/{exp_id}/save_models'
    try:
        os.mkdir(save_model_dir)
    except Exception as e:
        print(e)

    # ==== INIT MODEL=================
    device = get_device()

    model.to(device)
    optimizer = Ranger(model.parameters(),
                       lr=cfg["model_params"]["lr"],
                       weight_decay=0.0001)

    
    train_dataloader, gt_dict = load_tune_data()

    start = time.time()

    train_result = {
        'epoch': [],
        'iter': [],
        'loss[-k:](avg)': [],
        'validation_loss': [],
        'time(minute)': [],
    }
    def append_train_result(epoch, iter, avg_lossk, validation_loss, run_time):
        train_result['epoch'].append(epoch)
        train_result['iter'].append(iter)
        train_result['loss[-k:](avg)'].append(avg_lossk)
        train_result['validation_loss'].append(validation_loss)
        train_result['time(minute)'].append(run_time)


    k = 1000
    lossk = []
    torch.set_grad_enabled(True)
    num_iter = len(train_dataloader)
    print(num_iter)
    for epoch in range(cfg['train_params']['epoch']):
        model.train()
        torch.set_grad_enabled(True)

        tr_it = iter(train_dataloader)
        train_progress_bar = tqdm(range(num_iter))
        optimizer.zero_grad()
        print('epoch:', epoch)
        for i in train_progress_bar:
            try:
                data = next(tr_it)
                preds, confidences = forward(data, model, device)

                # convert to world positions
                world_from_agents = data['world_from_agent'].float().to(device)
                centroids = data['centroid'].float().to(device)
                world_from_agents = world_from_agents.unsqueeze(1) # bs * 1 * 3 * 3
                world_from_agents = world_from_agents.repeat(1, 3, 1, 1) # bs * 1 * 3 * 3
                centroids = centroids.unsqueeze(1).unsqueeze(1)
                centroids = centroids.repeat(1, 3, 50, 1)
                preds = transform_ts_points(preds, world_from_agents.clone()) - centroids[:, :, :, :2].clone()
                
                # get ground_truth
                target_availabilities = []
                target_positions = []
                for track_id, timestamp in zip(data['track_id'], data['timestamp']):
                    key = str(track_id.item()) + str(timestamp.item())
                    target_positions.append(torch.tensor(gt_dict[key]['coord']))
                    target_availabilities.append(torch.tensor(gt_dict[key]['avail']))
                
                target_availabilities = torch.stack(target_availabilities).to(device)
                target_positions = torch.stack(target_positions).to(device)

                loss = criterion(target_positions, preds, confidences, target_availabilities)
                
                # Backward pass
                optimizer.zero_grad()
                
                loss.backward()
                optimizer.step()

                lossk.append(loss.item())
                if len(lossk) > k:
                    lossk.pop(0)

                train_progress_bar.set_description(
                    f"loss: {loss.item():.4f} loss[-k:](avg): {np.mean(lossk):.4f}")

                if ((i > 0 and i % cfg['train_params']['checkpoint_steps'] == 0) or i == num_iter-1):
                    # save model per checkpoint
                    torch.save(model.state_dict(),
                               f'{save_model_dir}/epoch{epoch:02d}_iter{i:05d}.pth')
                    append_train_result(epoch, i, np.mean(lossk), -1, (time.time()-start)/60)


                if ((i > 0 and i % cfg['train_params']['validation_steps'] == 0 )
                        or i == num_iter-1):
                    validation_loss = validation(model, device)
                    append_train_result(epoch, i, -1, validation_loss, (time.time()-start)/60)
                    model.train()
                    torch.set_grad_enabled(True)

            except KeyboardInterrupt:
                torch.save(model.state_dict(),
                           f'{save_model_dir}/interrupt_epoch{epoch:02d}_iter{i:05d}.pth')
                # save train result
                results = pd.DataFrame(train_result)
                results.to_csv(
                    f"experiment/{exp_id}/interrupt_train_result.csv", index=False)
                print(f"Total training time is {(time.time()-start)/60} mins")
                print(results)
                raise KeyboardInterrupt
        
        # torch.save(model.state_dict(), f'{save_model_dir}/epoch{epoch:02d}_iter{i:05d}.pth')
        del tr_it, train_progress_bar

    # save train result
    results = pd.DataFrame(train_result)
    results.to_csv(f"experiment/{exp_id}/train_result.csv", index=False)
    print(f"Total training time is {(time.time()-start)/60} mins")
    print(results)


def inference(model, exp_id):
    device = get_device()
    # predict
    model.eval()
    torch.set_grad_enabled(False)
    test_dataloader = load_test_data()
    # store information for evaluation
    future_coords_offsets_pd = []
    timestamps = []
    confidences_list = []
    agent_ids = []

    progress_bar = tqdm(test_dataloader)
    with torch.no_grad():
        for data in progress_bar:
            preds, confidences = forward(data, model, device)

            # fix for the new environment
            preds = preds.cpu().numpy()
            world_from_agents = data["world_from_agent"].numpy()
            centroids = data["centroid"].numpy()
            coords_offset = []

            # convert into world coordinates and compute offsets
            for idx in range(len(preds)):
                for mode in range(3):
                    preds[idx, mode, :, :] = transform_points(
                        preds[idx, mode, :, :], world_from_agents[idx]) - centroids[idx][:2]

            future_coords_offsets_pd.append(preds.copy())
            confidences_list.append(confidences.cpu().numpy().copy())
            timestamps.append(data["timestamp"].numpy().copy())
            agent_ids.append(data["track_id"].numpy().copy())

    # create submission to submit to Kaggle
    pred_path = f'experiment/{exp_id}/submission.csv'
    write_pred_csv(pred_path,
                   timestamps=np.concatenate(timestamps),
                   track_ids=np.concatenate(agent_ids),
                   coords=np.concatenate(future_coords_offsets_pd),
                   confs=np.concatenate(confidences_list)
                   )


def main():
    args = parse_args()
    initialize(args.exp_id)
    model = get_model(cfg)
    train(model, args.exp_id)
    inference(model, args.exp_id)


if __name__ == "__main__":
    main()
