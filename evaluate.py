import os
import json
import torch
import numpy as np
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader

from datasets.tusimple import TuSimpleLaneDataset
from models.ultrafast_lane_net import UltraFastLaneNet
from utils.dataloader_utils import lane_collate_fn
from utils.visualize_labels import visualize_cls_targets

@torch.no_grad()
def softmax_argmax(pred_logits, ignore_index):
    """
    Convert logits to class prediction (grid index), excluding ignore_index.
    """
    probs = torch.softmax(pred_logits[:, :, :, :ignore_index], dim=-1)
    pred = torch.argmax(probs, dim=-1)
    return pred  # [B, L, R]

def match_accuracy(preds, gts, row_anchors, threshold=20):
    """
    Match predicted lanes to ground truth lanes for accuracy computation.
    """
    correct, total = 0, 0
    for pred_lanes, gt_lanes in zip(preds, gts):
        for gt_lane in gt_lanes:
            for (gt_x, y) in gt_lane:
                total += 1
                matched = False
                for pred_lane in pred_lanes:
                    for (pred_x, pred_y) in pred_lane:
                        if abs(y - pred_y) < 1e-3 and abs(gt_x - pred_x) <= threshold:
                            correct += 1
                            matched = True
                            break
                    if matched:
                        break
    return correct / total if total > 0 else 0.0

def evaluate():
    # Paths and parameters
    model_path = "checkpoints/best_model.pth"
    val_json = "TUSimple/test_set/val_label.json"
    dataset_root = "TUSimple/test_set"
    img_size = (360, 640)
    gridding_num = 100
    num_lanes = 4
    num_rows = 56
    row_anchors = np.linspace(160, 710, num_rows).astype(int).tolist()
    ignore_index = gridding_num

    batch_size = 8
    visualize_limit = 30
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Dataset & Dataloader
    dataset = TuSimpleLaneDataset(
        json_file=val_json,
        dataset_root=dataset_root,
        img_size=img_size,
        gridding_num=gridding_num,
        num_lanes=num_lanes,
        row_anchors=row_anchors
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lane_collate_fn)

    # Model
    model = UltraFastLaneNet(
        backbone="resnet18",
        pretrained=False,
        gridding_num=gridding_num,
        num_rows=num_rows,
        num_lanes=num_lanes
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # Evaluation
    all_preds, all_gts = [], []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            images = batch['image']
            targets = batch['cls_targets']
            img_paths = batch['img_path']
            lane_points = batch['lanes']

            logits = model(images)
            preds = softmax_argmax(logits, ignore_index)  # [B, L, R]

            # Convert predictions to (x, y)
            col_sample = np.linspace(0, img_size[1] - 1, gridding_num)
            batch_pred_lanes = []
            for b in range(preds.shape[0]):
                lanes = []
                for l in range(num_lanes):
                    points = []
                    for r, grid_idx in enumerate(preds[b, l]):
                        if grid_idx == ignore_index:
                            continue
                        x = int(col_sample[grid_idx])
                        y = int(row_anchors[r] * img_size[0] / 720)
                        points.append((x, y))
                    if points:
                        lanes.append(points)
                batch_pred_lanes.append(lanes)

                # Optional visualization
                if i * batch_size + b < visualize_limit:
                    out_path = os.path.join(output_dir, f"pred_{i * batch_size + b}.png")
                    visualize_cls_targets(
                        image_tensor=images[b],
                        cls_targets=preds[b],
                        row_anchors=row_anchors,
                        gridding_num=gridding_num,
                        img_path=img_paths[b],
                        save_path=out_path
                    )

            all_preds.extend(batch_pred_lanes)
            all_gts.extend(lane_points)

    acc = match_accuracy(all_preds, all_gts, row_anchors)
    print(f"\nTuSimple Accuracy: {acc * 100:.2f}%")

if __name__ == "__main__":
    evaluate()