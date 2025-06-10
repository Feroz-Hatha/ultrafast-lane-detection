import numpy as np

def generate_lane_targets(lane_points, row_anchors, img_w, gridding_num, num_lanes=4):
    """
    Converts continuous (x, y) points into grid-based classification targets.
    
    Args:
        lane_points: List of List of (x, y) points per lane
        row_anchors: List of y-values where predictions are made (fixed row anchors)
        img_w: Image width (e.g., 640)
        gridding_num: Number of horizontal bins (e.g., 100)
        num_lanes: Max number of lanes per image
        
    Returns:
        cls_targets: shape [num_lanes, len(row_anchors)] with column indices (0 to griding_num-1 or -1 for no lane)
    """
    cls_targets = np.ones((num_lanes, len(row_anchors)), dtype=np.int64) * gridding_num  # gridding_num (100) = no lane

    # Bin width in pixels
    col_sample = np.linspace(0, img_w - 1, gridding_num)
    col_sample_w = col_sample[1] - col_sample[0]

    for lane_idx, lane in enumerate(lane_points):
        if lane_idx >= num_lanes:
            break

        for x, y in lane:
            # Find the closest anchor row
            row_diff = [abs(y - r) for r in row_anchors]
            min_idx = np.argmin(row_diff)

            if abs(y - row_anchors[min_idx]) > 5:  # Optional: skip if too far off
                continue

            # Map x to grid index
            grid_idx = int(x / col_sample_w)
            if 0 <= grid_idx < gridding_num:
                cls_targets[lane_idx, min_idx] = grid_idx

    return cls_targets  # shape: [num_lanes, num_rows]