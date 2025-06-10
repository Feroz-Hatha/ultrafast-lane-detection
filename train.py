import os
import yaml
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from datasets.tusimple import TuSimpleLaneDataset
from models.ultrafast_lane_net import UltraFastLaneNet
from utils.dataloader_utils import lane_collate_fn
from utils.losses import structure_aware_loss

def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_row_anchors(start, end, num_rows):
    return list(map(int, torch.linspace(start, end, num_rows).tolist()))

def train():
    config = load_config()

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Row anchors
    row_anchors = get_row_anchors(
        config['dataset']['row_anchors'][0],
        config['dataset']['row_anchors'][1],
        config['dataset']['num_rows']
    )

    # Dataset & DataLoader
    dataset = TuSimpleLaneDataset(
        json_file=config['dataset']['json_file'],
        dataset_root=config['dataset']['dataset_root'],
        img_size=tuple(config['dataset']['img_size']),
        gridding_num=config['dataset']['gridding_num'],
        num_lanes=config['dataset']['num_lanes'],
        row_anchors=row_anchors
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        collate_fn=lane_collate_fn
    )

    # Model
    model = UltraFastLaneNet(
        backbone=config['model']['backbone'],
        pretrained=config['model']['pretrained'],
        gridding_num=config['dataset']['gridding_num'],
        num_rows=config['dataset']['num_rows'],
        num_lanes=config['dataset']['num_lanes']
    ).to(device)

    ignore_index = config['dataset']['gridding_num']
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['train']['lr'],
        weight_decay=config['train']['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['train']['epochs'])

    model.train()
    os.makedirs(config['train']['save_dir'], exist_ok=True)

    best_loss = float('inf')
    loss_history = {'total': [], 'cls': [], 'str': []}

    for epoch in range(config['train']['epochs']):
        total_loss, total_cls, total_str = 0.0, 0.0, 0.0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['train']['epochs']}", unit="batch")

        for batch in loop:
            images = batch['image'].to(device)
            targets = batch['cls_targets'].to(device)

            outputs = model(images)
            B, L, R, G_plus_1 = outputs.shape

            loss_cls = criterion(
                outputs.view(B * L * R, G_plus_1),
                targets.view(B * L * R)
            )

            loss_str = structure_aware_loss(
                logits=outputs,
                gridding_num=config['dataset']['gridding_num'],
                ignore_index=ignore_index,
                lambda_shp=config['train']['lambda_shp']
            )

            loss = loss_cls + config['train']['lambda_structure'] * loss_str

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_cls += loss_cls.item()
            total_str += loss_str.item()

            loop.set_postfix(cls=loss_cls.item(), str=loss_str.item(), total=loss.item())

        scheduler.step()

        avg_loss = total_loss / len(dataloader)
        avg_cls = total_cls / len(dataloader)
        avg_str = total_str / len(dataloader)

        loss_history['total'].append(avg_loss)
        loss_history['cls'].append(avg_cls)
        loss_history['str'].append(avg_str)

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = os.path.join(config['train']['save_dir'], "best_model.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved best model at epoch {epoch+1} with loss {best_loss:.4f}")

    # Plot loss
    plt.figure(figsize=(8, 6))
    plt.plot(loss_history['total'], label='Total Loss')
    plt.plot(loss_history['cls'], label='Classification Loss')
    plt.plot(loss_history['str'], label='Structure Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(config['train']['save_dir'], "loss_curve.png"))
    plt.close()

if __name__ == "__main__":
    train()