import torch
import torch.nn.functional as F

def structure_aware_loss(logits, gridding_num, ignore_index, lambda_shp=0.15):
    """
    Compute structure-aware loss as described in the original paper.

    Args:
        logits: Tensor of shape [B, C, R, G+1] (G = gridding_num)
        gridding_num: Number of horizontal grid bins
        ignore_index: Index corresponding to 'no lane'
        lambda_shp: Weight for shape loss

    Returns:
        Scalar loss (L_sim + lambda_shp * L_shp)
    """

    B, C, R, G_plus_1 = logits.shape
    G = gridding_num

    # ------------------------
    # 1. Similarity Loss (L1 between adjacent logits)
    # ------------------------
    sim_loss = F.l1_loss(logits[:, :, :-1, :], logits[:, :, 1:, :])

    # ------------------------
    # 2. Shape Loss (2nd-order diff of expected positions)
    # ------------------------

    # Exclude "no-lane" class from softmax (G-th index)
    logits_valid = logits[:, :, :, :G]  # [B, C, R, G]
    prob = F.softmax(logits_valid, dim=-1)  # [B, C, R, G]

    grid = torch.arange(G, dtype=logits.dtype, device=logits.device)
    loc = (prob * grid).sum(dim=-1)  # [B, C, R]

    # Compute curvature = second-order difference
    dx1 = loc[:, :, 1:-1] - loc[:, :, :-2]    # [B, C, R-2]
    dx2 = loc[:, :, 2:] - loc[:, :, 1:-1]     # [B, C, R-2]
    ddx = dx2 - dx1                           # [B, C, R-2]

    shp_loss = ddx.abs().mean()

    return sim_loss + lambda_shp * shp_loss