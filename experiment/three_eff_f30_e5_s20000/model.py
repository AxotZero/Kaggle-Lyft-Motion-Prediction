import torch
from torch import nn
from efficientnet_pytorch import model as enet


# Implementation of class to load the particular EfficientNet model.
class LyftModel(torch.nn.Module):
    def __init__(self, cfg, num_modes=3):
        super().__init__()
        self.eff1 = EfficientnetBlock(cfg)
        self.eff2 = EfficientnetBlock(cfg)
        self.eff3 = EfficientnetBlock(cfg)
    
    def forward(self, x):
        """Function to perform forward propagation."""
        conf1, pred1 = self.eff1(x.clone())
        conf2, pred2 = self.eff2(x.clone())
        conf3, pred3 = self.eff3(x.clone())
        
        confidences = torch.cat((conf1, conf2, conf3), dim=1)
        preds = torch.cat((pred1, pred2, pred3), dim=1)
        
        confidences = torch.softmax(confidences, dim=1)

        return preds, confidences


class EfficientnetBlock(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = enet.EfficientNet.from_name(cfg["model_params"]["model_architecture"])
        
        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels

        self.num_preds = 2 * cfg["model_params"]["future_num_frames"]
        num_targets = 1 + self.num_preds
    
        self.backbone._conv_stem = nn.Conv2d(
            num_in_channels,
            self.backbone._conv_stem.out_channels,
            kernel_size=self.backbone._conv_stem.kernel_size,
            stride=self.backbone._conv_stem.stride,
            padding=self.backbone._conv_stem.padding,
            bias=False
        )
    
        self.backbone._fc = nn.Linear(in_features=self.backbone._fc.in_features, out_features=num_targets)
    


    def forward(self, x):
        """Function to perform forward propagation."""
        x = self.backbone(x)
        preds, confidences  = torch.split(x, self.num_preds, dim=1)
        preds = preds.view(x.shape[0], 1, 50, 2)
        return confidences, preds


def load_pretrained(model, cfg):
    weight_path = cfg["model_params"]["weight_path"]
    model.load_state_dict(torch.load(weight_path), strict=True)


# def load_pretrained(model, cfg):
#     weight_path = cfg["model_params"]["weight_path"]

#     # replace layer name
#     pretrained_dict = torch.load(weight_path)
#     for key in list(pretrained_dict.keys()):
#         new_key = key.replace('backbone.', '')
#         pretrained_dict[new_key] = pretrained_dict.pop(key)

#     # delete mismatch layer
#     ignore_keys = cfg["model_params"]["ignore_weight_keys"]
#     pretrained_dict = {k: v for k, v in pretrained_dict.items() if not(k in ignore_keys)}

#     model.eff1.backbone.load_state_dict(pretrained_dict, strict=False)
#     model.eff2.backbone.load_state_dict(pretrained_dict, strict=False)
#     model.eff3.backbone.load_state_dict(pretrained_dict, strict=False)



def forward(data, model, device, criterion):
    inputs = data["image"].to(device)
    target_availabilities = data["target_availabilities"].to(device)
    targets = data["target_positions"].to(device)
    # Forward pass
    preds, confidences = model(inputs)
    loss = criterion(targets, preds, confidences, target_availabilities)
    return loss, preds, confidences


if __name__ == "__main__":
    pass