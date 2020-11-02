import torch
from torch import nn
from efficientnet_pytorch import model as enet
from lambda_networks import LambdaLayer


# Implementation of class to load the particular EfficientNet model.
class LyftModel(torch.nn.Module):
    def __init__(self, cfg, num_modes=3):
        super().__init__()
        self.backbone = enet.EfficientNet.from_name(cfg["model_params"]["model_architecture"])
        
        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels
        num_targets = 2*cfg["model_params"]["future_num_frames"]
        self.future_len = cfg["model_params"]["future_num_frames"]
        
        h = w = cfg["raster_params"]["raster_size"][0] // 4
        self.backbone._conv_stem = LambdaLayer(
            dim = num_in_channels,       # channels going in
            dim_out = self.backbone._conv_stem.out_channels,   # channels out
            n = h * w,    # number of input pixels (64 x 64 image)
            dim_k = 16,     # key dimension
            heads = 4,      # number of heads, for multi-query
            dim_u = 4     # 'intra-depth' dimension
        )

        # self.backbone._conv_stem = nn.Conv2d(
        #     num_in_channels,
        #     self.backbone._conv_stem.out_channels,
        #     kernel_size=self.backbone._conv_stem.kernel_size,
        #     stride=self.backbone._conv_stem.stride,
        #     padding=self.backbone._conv_stem.padding,
        #     bias=False
        # )

        self.num_preds = num_targets * num_modes
        self.num_modes = num_modes

        self.backbone._fc = nn.Linear(in_features=self.backbone._fc.in_features, 
                                        out_features=self.num_preds + num_modes)
    
    def forward(self, x):
        """Function to perform forward propagation."""
        x = self.backbone(x)

        bs, _ = x.shape
        pred, confidences = torch.split(x, self.num_preds, dim=1)
        pred = pred.view(bs, self.num_modes, self.future_len, 2)
        assert confidences.shape == (bs, self.num_modes)
        confidences = torch.softmax(confidences, dim=1)

        return pred, confidences


def forward(data, model, device, criterion):
    inputs = data["image"].to(device)
    target_availabilities = data["target_availabilities"].to(device)
    targets = data["target_positions"].to(device)
    # Forward pass
    preds, confidences = model(inputs)
    loss = criterion(targets, preds, confidences, target_availabilities)
    return loss, preds, confidences
