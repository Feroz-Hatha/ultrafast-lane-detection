import torch
import torch.nn as nn
import torchvision.models as models


class UltraFastLaneNet(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=True, gridding_num=100, num_rows=56, num_lanes=4):
        super(UltraFastLaneNet, self).__init__()

        self.gridding_num = gridding_num
        self.num_rows = num_rows
        self.num_lanes = num_lanes

        # Load backbone (ResNet-18) and remove its final FC and pooling layers
        resnet = models.resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )

        # Project features to num_rows * num_lanes * gridding_num
        # Output shape: [B, num_lanes * num_rows * gridding_num, H', W']
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_lanes * num_rows * (gridding_num + 1), kernel_size=1)
        )

    def forward(self, x):
        feat = self.backbone(x)                             # [B, 512, H', W']
        out = self.classifier(feat)                         # [B, L*R*G, H', W']
        B = x.size(0)
        out = out.mean(dim=2).mean(dim=2)                   # Global average pool → [B, L*R*G]

        # Reshape to [B, num_lanes, num_rows, gridding_num]
        out = out.view(B, self.num_lanes, self.num_rows, self.gridding_num + 1)
        return out