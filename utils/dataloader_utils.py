import torch

def lane_collate_fn(batch):
    """
    Custom collate function for TuSimple Lane Detection.
    This function:
    - Stacks images and classification targets into tensors
    - Keeps lane annotations and image paths as lists (variable-length)

    Returns:
        {
            'image': Tensor of shape [B, 3, H, W],
            'cls_targets': Tensor of shape [B, num_lanes, num_rows],
            'lanes': list of lane point lists (variable lengths),
            'img_path': list of strings
        }
    """
    images = torch.stack([item['image'] for item in batch])                   # [B, 3, H, W]
    cls_targets = torch.stack([item['cls_targets'] for item in batch])       # [B, num_lanes, num_rows]
    lanes = [item['lanes'] for item in batch]                                # List[List[(x, y)]]
    img_paths = [item['img_path'] for item in batch]                         # List[str]

    return {
        'image': images,
        'cls_targets': cls_targets,
        'lanes': lanes,
        'img_path': img_paths
    }