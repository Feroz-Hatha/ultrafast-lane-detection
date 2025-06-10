import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_cls_targets(image_tensor, cls_targets, row_anchors, gridding_num, img_path=None, save_path=None):
    """
    Visualize classification targets (cls_targets) over the image.

    Args:
        image_tensor: Torch tensor [3, H, W] (normalized)
        cls_targets: Tensor [num_lanes, num_rows]
        row_anchors: List of y-values (e.g., 56 values)
        griding_num: Number of horizontal bins (e.g., 100)
        img_path: Optional path (for debug title)
        save_path: If provided, saves the visualization
    """
    # Undo normalization and convert to numpy
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
    image_np = image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  # unnormalize
    image_np = np.clip(image_np, 0, 1)
    image_np = (image_np * 255).astype(np.uint8)
    image_np = np.ascontiguousarray(image_np)  # <-- REQUIRED for OpenCV
    
    img_h, img_w, _ = image_np.shape
    col_sample = np.linspace(0, img_w - 1, gridding_num)

    # Plot each valid (x, y) for each lane
    for lane_idx, lane in enumerate(cls_targets):
        for row_idx, grid_x in enumerate(lane):
            # if grid_x == -1:
            if grid_x < 0 or grid_x >= gridding_num:
                continue
            x = int(col_sample[grid_x])
            y = int(row_anchors[row_idx] * img_h / 720)  # scale anchors to resized image height
            cv2.circle(image_np, (x, y), radius=4, color=(0, 255, 0), thickness=-1)

    # Show using matplotlib
    plt.figure(figsize=(10, 5))
    if img_path:
        plt.title(f"Visualization: {img_path.split('/')[-1]}")
    plt.imshow(image_np)
    plt.axis('off')

    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    else:
        plt.show()