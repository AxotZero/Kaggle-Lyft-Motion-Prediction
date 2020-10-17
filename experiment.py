import os
import random
import time
import argparse
from tqdm import tqdm

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
from l5kit.evaluation import write_pred_csv
from l5kit.evaluation.metrics import neg_multi_log_likelihood
from l5kit.geometry import transform_points


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp_id', type=str, default='', required=True,
                        help='path of your experiment directory name')
    return parser.parse_args()


def initialize(exp_id):

    seed = int(time.time())
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    # load experiment config and model architecture
    module_path = f'experiment.{exp_id}'
    exec(f'from {module_path}.config import *', globals()) 
    exec(f'from {module_path}.model import LyftModel', globals())
    exec(f'from {module_path}.model import forward', globals())
    # print('GPU=', GPU)
    

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


def load_train_data():
    # load training data
    dm = get_dm()
    train_cfg = cfg["train_data_loader"]

    rasterizer = build_rasterizer(cfg, dm)
    train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()
    train_dataset = AgentDataset(cfg, train_zarr, rasterizer)
    train_dataloader = DataLoader(train_dataset, 
                                shuffle=train_cfg["shuffle"],
                                batch_size=train_cfg["batch_size"], 
                                num_workers=train_cfg["num_workers"])
    print('len(train_dataloader):', len(train_dataloader))
    return train_dataloader


def load_test_data():
    dm = get_dm()
    test_cfg = cfg["test_data_loader"]
    rasterizer = build_rasterizer(cfg, dm)
    test_zarr = ChunkedDataset(dm.require(test_cfg["key"])).open()
    test_mask = np.load(f'{cfg["data_path"]}/scenes/mask.npz')["arr_0"]
    test_dataset = AgentDataset(cfg, test_zarr, rasterizer, agents_mask=test_mask)
    test_dataloader = DataLoader(test_dataset,
                                shuffle=test_cfg["shuffle"],
                                batch_size=test_cfg["batch_size"],
                                num_workers=test_cfg["num_workers"])
    return test_dataloader


def get_model(cfg):
    model = LyftModel(cfg)
    #load weight if there is a pretrained model
    weight_path = cfg["model_params"]["weight_path"]
    if weight_path:
        model.load_state_dict(torch.load(weight_path))

    # print('cfg = ', cfg)
    # print('model =', model)


    return model


def criterion(
    gt: Tensor, pred: Tensor, confidences: Tensor, avails: Tensor
) -> Tensor:
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

    assert gt.shape == (batch_size, future_len, num_coords), f"expected 2D (Time x Coords) array for gt, got {gt.shape}"
    assert confidences.shape == (batch_size, num_modes), f"expected 1D (Modes) array for gt, got {confidences.shape}"

    # assert torch.allclose(torch.sum(confidences, dim=1) , 
    #                         confidences.new_ones((batch_size,)) ), "confidences should sum to 1"
    if not (torch.allclose(torch.sum(confidences, dim=1) , confidences.new_ones((batch_size,)) )):
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
    error = torch.sum(((gt - pred) * avails) ** 2, dim=-1)  # reduce coords and use availability

    with np.errstate(divide="ignore"):  # when confidence is 0 log goes to -inf, but we're fine with it
        # error (batch_size, num_modes)
        error = torch.log(confidences) - 0.5 * torch.sum(error, dim=-1)  # reduce time

    # use max aggregator on modes for numerical stability
    # error (batch_size, num_modes)
    max_value, _ = error.max(dim=1, keepdim=True)  # error are negative at this point, so max() gives the minimum one
    error = -torch.log(torch.sum(torch.exp(error - max_value), dim=-1, keepdim=True)) - max_value  # reduce modes
    # print("error", error)
    return torch.mean(error)


def train(model, exp_id):
    save_model_dir = f'experiment/{exp_id}/save_models'
    try:
        os.mkdir(save_model_dir)
    except Exception as e :
        print(e)

    # ==== INIT MODEL=================
    device = get_device()
    
    model.to(device)
    # optimizer = optim.Adam(model.parameters(), lr=cfg["model_params"]["lr"])
    optimizer = Ranger(model.parameters(), lr=cfg["model_params"]["lr"])
    train_dataloader = load_train_data()
    start = time.time()

    train_result = {
        'epoch': [],
        'iter': [],
        'loss': [],
        'checkpoint_loss(avg)': [],
        'total_loss(avg)': [],
        'time(minute)': [],
    }
    
    total_loss = []
    checkpoint_loss = []

    model.train()
    torch.set_grad_enabled(True)

    for epoch in range(cfg['train_params']['epoch']):
        tr_it = iter(train_dataloader)
        num_iter = cfg["train_params"]["max_num_steps"]
        progress_bar = tqdm(range(num_iter))

        for i in progress_bar:
            data = next(tr_it)
            loss, _, _ = forward(data, model, device, criterion)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.append(loss.item())
            checkpoint_loss.append(loss.item())
            progress_bar.set_description(f"loss: {loss.item()} checkpoint_loss(avg): {np.mean(checkpoint_loss)}")

            if ((i > 0 and i % cfg['train_params']['checkpoint_steps'] == 0) 
                or i == num_iter-1):
                # save model per checkpoint
                torch.save(model.state_dict(), 
                            f'{save_model_dir}/epoch{epoch:02d}_iter{i:05d}.pth')

                train_result['epoch'].append(epoch)
                train_result['iter'].append(i)
                train_result['loss'].append(loss.item())
                train_result['checkpoint_loss(avg)'].append(np.mean(checkpoint_loss))
                train_result['total_loss(avg)'].append(np.mean(total_loss))
                train_result['time(minute)'].append((time.time()-start)/60)

                checkpoint_loss = []


    # save train result
    results = pd.DataFrame(train_result)
    results.to_csv(f"experiment/{exp_id}/train_result.csv", index=False)
    print(f"Total training time is {(time.time()-start)/60} mins")
    print(results.head())


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

    for data in progress_bar:
        
        _, preds, confidences = forward(data, model, device, criterion)

        #fix for the new environment
        preds = preds.cpu().numpy()
        world_from_agents = data["world_from_agent"].numpy()
        centroids = data["centroid"].numpy()
        coords_offset = []
        
        # convert into world coordinates and compute offsets
        for idx in range(len(preds)):
            for mode in range(3):
                preds[idx, mode, :, :] = transform_points(preds[idx, mode, :, :], world_from_agents[idx]) - centroids[idx][:2]

        future_coords_offsets_pd.append(preds.copy())
        confidences_list.append(confidences.cpu().numpy().copy())
        timestamps.append(data["timestamp"].numpy().copy())
        agent_ids.append(data["track_id"].numpy().copy()) 

    #create submission to submit to Kaggle
    pred_path = f'experiment/{exp_id}/submission.csv'
    write_pred_csv(pred_path,
            timestamps=np.concatenate(timestamps),
            track_ids=np.concatenate(agent_ids),
            coords=np.concatenate(future_coords_offsets_pd),
            confs = np.concatenate(confidences_list)
            )

def main():
    args = parse_args()
    initialize(args.exp_id)
    model = get_model(cfg)
    train(model, args.exp_id)
    inference(model, args.exp_id)


if __name__ == "__main__":
    main()