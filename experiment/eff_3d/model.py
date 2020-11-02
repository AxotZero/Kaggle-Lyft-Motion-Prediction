import torch
from torch import nn
from efficientnet_pytorch_3d import EfficientNet3D
import numpy as np

# Implementation of class to load the particular EfficientNet model.


class LyftModel(torch.nn.Module):
    def __init__(self, cfg, num_modes=3):
        super().__init__()

        # num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        # num_in_channels = 3 + num_history_channels
        # num_targets = 2*cfg["model_params"]["future_num_frames"]

        self.frame_num = cfg["model_params"]["history_num_frames"] + 1

        # self.future_len = cfg["model_params"]["future_num_frames"]

        # self.backbone._conv_stem = nn.Conv2d(
        #     num_in_channels,
        #     self.backbone._conv_stem.out_channels,
        #     kernel_size=self.backbone._conv_stem.kernel_size,
        #     stride=self.backbone._conv_stem.stride,
        #     padding=self.backbone._conv_stem.padding,
        #     bias=False
        # )

        # self.sementic_conv = nn.Conv2d(
        #     in_channels=3,
        #     out_channels=1,
        #     kernel_size=(3, 3),
        #     stride=1,
        #     padding=1,
        #     bias=True
        # )

        self.single_output = 101
        self.multi_output = 303
        
        self.backbone = EfficientNet3D.from_name(cfg["model_params"]["model_architecture"],
                                                 in_channels=(5))

        self.fc1 = nn.Linear(
            in_features=self.backbone._fc.in_features,
            out_features=self.single_output
        )

        self.fc2 = nn.Linear(
            in_features=self.backbone._fc.in_features,
            out_features=self.single_output
        )
        self.fc3 = nn.Linear(
            in_features=self.backbone._fc.in_features,
            out_features=self.single_output
        )

        self.backbone._fc = nn.ModuleDict({
            'fc1': self.fc1,
            'fc2': self.fc2,
            'fc3': self.fc3}
        )
        print(self)

    def backbone_forward(self, x):

        x = self.backbone.extract_features(x)
        x = self.backbone._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.backbone._dropout(x)
        y1 = self.backbone._fc['fc1'](x)
        y2 = self.backbone._fc['fc2'](x)
        y3 = self.backbone._fc['fc3'](x)
        return y1, y2, y3

    def split_output(self, x):
        pred, conf = torch.split(x, self.single_output-1, dim=1)
        pred = pred.view(x.shape[0], 1, 50, 2)
        return pred, conf

    def forward(self, agents, egos, map_sem):
        """
        Function to perform forward propagation.
        input:
            agents: (bs, 1, fn, h, w)
            map_sem: (bs, 3, h, w)
        """
        # map_sem = self.sementic_conv(map_sem)       # (bs, 1, h, w)
        map_sem = torch.unsqueeze(map_sem, dim=2)   # (bs, 1, 1, h, w)
        map_sem = map_sem.repeat(1, 1, self.frame_num,
                                 1, 1)  # (bs, 3, fn, h, w)

        backbone_inputs = torch.cat(
            (agents, egos, map_sem), dim=1)  # (bs, 2, fn, h, w)

        y1, y2, y3 = self.backbone_forward(backbone_inputs)

        pred1, conf1 = self.split_output(y1)
        pred2, conf2 = self.split_output(y2)
        pred3, conf3 = self.split_output(y3)

        confidences = torch.cat((conf1, conf2, conf3), dim=1)
        preds = torch.cat((pred1, pred2, pred3), dim=1)

        confidences = torch.softmax(confidences, dim=1)

        return preds, confidences


def forward(data, model, device, criterion):
    images = data["image"]
    total_frames_num = (images.shape[1] - 3) // 2

    # split
    agents = np.expand_dims(images[:, :total_frames_num], axis=1)   # bs * 1 * fn * h * w
    egos = np.expand_dims(images[:, total_frames_num:-3], axis=1)   # bs * 1 * fn * h * w
    # cars = agents / 2 + egos              # bs * fn * h * w

    # input of model
    agents = torch.Tensor(agents).to(device)  # bs * 1 * fn * h * w
    egos = torch.Tensor(egos).to(device)
    map_sem = torch.Tensor(images[:, -3:]).to(device) # bs * 3 * h * w

    # expand_dims

    # egos = np.expand_dims(egos, axis=1)         # bs * 1 * fn * h * w
    # map_sem = np.expand_dims(map_sem, axis=2)   # bs * 3 * 1  * h * w
    # map_sem = np.tile(map_sem, [1, 1, total_frames_num, 1, 1])

    # get inputs
    # inputs = torch.Tensor(np.concatenate((agents, egos, map_sem), axis=1)).to(device)

    target_availabilities = data["target_availabilities"].to(device)
    targets = data["target_positions"].to(device)

    # Forward pass
    preds, confidences = model(agents, egos, map_sem)
    loss = criterion(targets, preds, confidences, target_availabilities)
    return loss, preds, confidences


# def forward(data, model, device, criterion):
#     inputs = data["image"].to(device)
#     target_availabilities = data["target_availabilities"].to(device)
#     targets = data["target_positions"].to(device)
#     # Forward pass
#     preds, confidences = model(inputs)
#     loss = criterion(targets, preds, confidences, target_availabilities)
#     return loss, preds, confidences
