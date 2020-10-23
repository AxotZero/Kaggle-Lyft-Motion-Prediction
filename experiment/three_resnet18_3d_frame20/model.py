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


        model_depth = cfg["model_params"]["model_depth"]
        self.backbone1 = generate_model(model_depth=model_depth,
                                        n_classes=(self.out_features//3),
                                        n_input_channels=frame_channels)
        self.backbone2 = generate_model(model_depth=model_depth,
                                        n_classes=(self.out_features//3),
                                        n_input_channels=frame_channels)
        self.backbone3 = generate_model(model_depth=model_depth,
                                        n_classes=(self.out_features//3),
                                        n_input_channels=frame_channels)
        

    def forward(self, x):
        """
            Function to perform forward propagation.
            x.shape = (bs, frames, 5, H, W)
        """
        batch_size = x.shape[0]
        output1 = self.backbone1(x.clone())
        output2 = self.backbone2(x.clone())
        output3 = self.backbone3(x.clone())

        conf1, pred1  = self.convert_backbone_output(output1)
        conf2, pred2  = self.convert_backbone_output(output2)
        conf3, pred3  = self.convert_backbone_output(output3)

        confidences = torch.cat((conf1, conf2, conf3), dim=1)
        preds = torch.cat((pred1, pred2, pred3), dim=1)
        confidences = torch.softmax(confidences, dim=1)

        return preds, confidences

    def convert_backbone_output(self, output):
        pred, conf  = torch.split(output, self.num_preds // 3, dim=1)
        pred = pred.view(output.shape[0], 1, 50, 2)
        return conf, pred

def load_pretrained(model, cfg):

    weight_path = cfg["model_params"]["weight_path"]
    model.load_state_dict(torch.load(weight_path))
    # ignore_keys = cfg["model_params"]["ignore_weight_keys"]


    # pretrained_dict = torch.load(weight_path)
    # for key in list(pretrained_dict.keys()):
    #     new_key = key.replace('backbone.', '')
    #     pretrained_dict[new_key] = pretrained_dict.pop(key)

    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if not(k in ignore_keys)}

    # model.backbone1.load_state_dict(pretrained_dict, strict=False)
    # model.backbone2.load_state_dict(pretrained_dict, strict=False)
    # model.backbone3.load_state_dict(pretrained_dict, strict=False)
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
