import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.target_generator import generate_lane_targets


class TuSimpleLaneDataset(Dataset):
    def __init__(self, json_file, dataset_root, img_size=(360, 640), gridding_num=100, num_lanes=4, row_anchors=None, transform=None):
        self.dataset_root = dataset_root
        self.img_size = img_size
        self.transform = transform
        self.gridding_num = gridding_num
        self.num_lanes = num_lanes

        # Define row anchors if not passed
        self.row_anchors = row_anchors or np.linspace(160, 710, 56).astype(int).tolist()

        # Load annotation JSON
        with open(json_file, 'r') as f:
            self.annotations = [json.loads(line) for line in f]

        # Define default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        sample = self.annotations[idx]

        # Load image
        img_path = os.path.join(self.dataset_root, sample['raw_file'])
        image = Image.open(img_path).convert('RGB')
        original_size = image.size  # (W, H)

        # Apply transform
        image = self.transform(image)

        # Parse lane annotations
        lanes = sample['lanes']
        h_samples = sample['h_samples']

        # Convert lane points to (x, y)
        lane_points = []
        for lane in lanes:
            points = []
            for x, y in zip(lane, h_samples):
                if x != -2:
                    norm_x = x / original_size[0] * self.img_size[1]
                    norm_y = y / original_size[1] * self.img_size[0]
                    points.append((norm_x, norm_y))
            lane_points.append(points)

        scaled_row_anchors = [r / original_size[1] * self.img_size[0] for r in self.row_anchors]

        # Convert to classification targets
        cls_targets = generate_lane_targets(
            lane_points=lane_points,
            row_anchors=scaled_row_anchors,
            img_w=self.img_size[1],
            gridding_num=self.gridding_num,
            num_lanes=self.num_lanes
        )

        return {
            'image': image,  # Tensor: [3, H, W]
            'cls_targets': torch.tensor(cls_targets, dtype=torch.long),  # [num_lanes, num_rows]
            'img_path': img_path,
            'lanes': lane_points
        }