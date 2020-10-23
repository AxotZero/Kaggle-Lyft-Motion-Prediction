import torch
from torch import nn
from efficientnet_pytorch import model as enet
import numpy as np

# Implementation of class to load the particular EfficientNet model.
class LyftModel(torch.nn.Module):
    def __init__(self, cfg, num_modes=3):
        super().__init__()
        frame_channels = 5

        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        num_targets = 2*cfg["model_params"]["future_num_frames"]

        self.future_len = cfg["model_params"]["future_num_frames"]
        self.num_frames = cfg["model_params"]["history_num_frames"] + 1
        self.num_preds = num_targets * num_modes
        self.num_modes = num_modes
        self.out_features = self.num_preds + self.num_modes

        backbone_out_features = 64
        self.backbone = EfficientNetBlock(cfg=cfg, num_in_channels=frame_channels, out_features=backbone_out_features)
        self.lstm = LSTMBlock(feature_nums=backbone_out_features, num_frames=self.num_frames)
        self.fc = nn.Linear(self.num_frames*backbone_out_features, self.out_features)

    def forward(self, x):
        """
            Function to perform forward propagation.
            x.shape = (bs, frames, 5, H, W)
        """
        batch_size = x.shape[0]
        
        batch_time_reprs = torch.cat([self.backbone(x[batch_idx]).unsqueeze(0) for batch_idx in range(batch_size)], 
                                        axis=0)
        # batch_time_reprs.shape = (bs, frames, 5, H, W) = (bs, frames, backbone_out_features)
        outputs = self.lstm(batch_time_reprs)

        outputs = self.fc(outputs)
        
        pred, confidences = torch.split(outputs, self.num_preds, dim=1)
        pred = pred.view(batch_size, self.num_modes, self.future_len, 2)
        assert confidences.shape == (batch_size, self.num_modes)
        confidences = torch.softmax(confidences, dim=1)

        return pred, confidences

class EfficientNetBlock(nn.Module):
    def __init__(self, cfg, num_in_channels=5, out_features=128):
        super().__init__()
        self.backbone = enet.EfficientNet.from_name(cfg["model_params"]["model_architecture"])
        self.backbone._conv_stem = nn.Conv2d(
            num_in_channels,
            self.backbone._conv_stem.out_channels,
            kernel_size=self.backbone._conv_stem.kernel_size,
            stride=self.backbone._conv_stem.stride,
            padding=self.backbone._conv_stem.padding,
            bias=False
        )

        self.backbone._fc = nn.Linear(in_features=self.backbone._fc.in_features, 
                                        out_features=out_features)
    def forward(self, x):
        return self.backbone(x)


class LSTMBlock(nn.Module):
    def __init__(self, feature_nums=128, num_frames=11):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size=feature_nums,
                            hidden_size=feature_nums,
                            num_layers=num_frames,
                            batch_first=True)

    def forward(self, input):
        """
        input.shape = (bs, frame_nums, eff_output)
        """
        input = input # shape = (frame_nums, bs, eff_output)
        
        output, (ht, ct) = self.lstm(input)
        output = torch.flatten(output, 1)
        return output


def forward(data, model, device, criterion):
    images = data["image"]
    total_frames_num = (images.shape[1] - 3) // 2

    # split
    agents = images[:, :total_frames_num]
    egos = images[:, total_frames_num:-3]
    map_sem = images[:, -3:]
    
    # expand_dims
    agents = np.expand_dims(agents, axis=2)
    egos = np.expand_dims(egos, axis=2)
    map_sem = np.expand_dims(map_sem, axis=1)
    map_sem = np.tile(map_sem, [1, total_frames_num, 1, 1, 1])

    # get inputs
    inputs = torch.Tensor(np.concatenate((agents, egos, map_sem), axis=2)).to(device)

    target_availabilities = data["target_availabilities"].to(device)
    targets = data["target_positions"].to(device)
    
    # Forward pass
    preds, confidences = model(inputs)
    loss = criterion(targets, preds, confidences, target_availabilities)
    return loss, preds, confidences
