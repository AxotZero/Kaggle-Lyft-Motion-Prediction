import torch
from torch import nn
from .resnet import generate_model
import numpy as np

class LyftModel(torch.nn.Module):
    def __init__(self, cfg, num_modes=3):
        super().__init__()
        frame_channels = 5

        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        num_targets = 2*cfg["model_params"]["future_num_frames"]

        self.future_len = cfg["model_params"]["future_num_frames"]
        # self.num_frames = cfg["model_params"]["history_num_frames"] + 1
        self.num_preds = num_targets * num_modes
        self.num_modes = num_modes
        self.out_features = self.num_preds + self.num_modes
        self.backbone = generate_model(model_depth=cfg["model_params"]["model_depth"],
                                        n_classes=self.out_features,
                                        n_input_channels=frame_channels)
        
        # self.backbone = EfficientNetBlock(cfg=cfg, in_channels=frame_channels, out_features=self.out_features)

    def forward(self, x):
        """
            Function to perform forward propagation.
            x.shape = (bs, frames, 5, H, W)
        """
        batch_size = x.shape[0]
        x = self.backbone(x)

        pred, confidences = torch.split(x, self.num_preds, dim=1)
        pred = pred.view(batch_size, self.num_modes, self.future_len, 2)
        assert confidences.shape == (batch_size, self.num_modes)
        confidences = torch.softmax(confidences, dim=1)

        return pred, confidences


def load_pretrained(model, weight_path):

    model.load_state_dict(torch.load(weight_path, map_location='cpu'))
    print(f'Load pretrained from {weight_path}')

    return model



def forward(data, model, device, criterion):
    images = data["image"]
    total_frames_num = (images.shape[1] - 3) // 2

    # split
    agents = images[:, :total_frames_num]   # bs * fn * h * w
    egos = images[:, total_frames_num:-3]   # bs * fn * h * w
    map_sem = images[:, -3:]                # bs * 3  * h * w
    
    # expand_dims
    agents = np.expand_dims(agents, axis=1)     # bs * 1 * fn * h * w
    egos = np.expand_dims(egos, axis=1)         # bs * 1 * fn * h * w
    map_sem = np.expand_dims(map_sem, axis=2)   # bs * 3 * 1  * h * w
    map_sem = np.tile(map_sem, [1, 1, total_frames_num, 1, 1])

    # get inputs
    inputs = torch.Tensor(np.concatenate((agents, egos, map_sem), axis=1)).to(device)

    target_availabilities = data["target_availabilities"].to(device)
    targets = data["target_positions"].to(device)
    
    # Forward pass
    preds, confidences = model(inputs)
    loss = criterion(targets, preds, confidences, target_availabilities)
    return loss, preds, confidences
