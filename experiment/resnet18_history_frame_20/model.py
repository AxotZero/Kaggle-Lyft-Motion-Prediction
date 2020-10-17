import torch
from torch import nn
from torchvision.models.resnet import resnet50, resnet18, resnet34, resnet101
# debug = True
class LyftModel(nn.Module):

    def __init__(self, cfg, num_modes=3):
        super().__init__()
        architecture = cfg["model_params"]["model_architecture"]
        backbone = eval(architecture)(pretrained=True, progress=True)
        self.backbone = backbone

        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels

        self.backbone.conv1 = nn.Conv2d(
            num_in_channels,
            self.backbone.conv1.out_channels,
            kernel_size=self.backbone.conv1.kernel_size,
            stride=self.backbone.conv1.stride,
            padding=self.backbone.conv1.padding,
            bias=True,
        )

        # This is 512 for resnet18 and resnet34;
        # And it is 2048 for the other resnets
        
        if architecture == "resnet50":
            backbone_out_features = 2048
        else:
            backbone_out_features = 512

        # X, Y coords for the future positions (output shape: batch_sizex50x2)
        self.future_len = cfg["model_params"]["future_num_frames"]
        num_targets = 2 * self.future_len

        # You can add more layers here.
        self.head = nn.Sequential(
            # nn.Dropout(0.2),
            nn.Linear(in_features=backbone_out_features, out_features=4096),
        )

        self.num_preds = num_targets * num_modes
        self.num_modes = num_modes

        self.logit = nn.Linear(4096, out_features=self.num_preds + num_modes)

    def forward(self, inputs):
        if not(torch.isfinite(inputs).all()): print('gggg')

        x = self.backbone.conv1(inputs)
        if not(torch.isfinite(x).all()): 
            print(1111)
            print(inputs)

        x = self.backbone.bn1(x)
        # if not(torch.isfinite(x).all()): print(2222)
        
        x = self.backbone.relu(x)
        # if not(torch.isfinite(x).all()): print(3333)

        x = self.backbone.maxpool(x)
        # if not(torch.isfinite(x).all()): print(4444)

        x = self.backbone.layer1(x)
        # if not(torch.isfinite(x).all()): print(5555)
        
        x = self.backbone.layer2(x)
        # if not(torch.isfinite(x).all()): print(6666)

        x = self.backbone.layer3(x)
        # if not(torch.isfinite(x).all()): print(7777)

        x = self.backbone.layer4(x)
        # if not(torch.isfinite(x).all()): print(8888)

        x = self.backbone.avgpool(x)
        # if not(torch.isfinite(x).all()): print(9999)

        x = torch.flatten(x, 1)

        x = self.head(x)
        # if not(torch.isfinite(x).all()): print(0000)
        
        x = self.logit(x)

        # pred (batch_size)x(modes)x(time)x(2D coords)
        # confidences (batch_size)x(modes)
        bs, _ = x.shape
        pred, confidences = torch.split(x, self.num_preds, dim=1)
        pred = pred.view(bs, self.num_modes, self.future_len, 2)
        assert confidences.shape == (bs, self.num_modes)

        # print('confidences before:', confidences) 
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
