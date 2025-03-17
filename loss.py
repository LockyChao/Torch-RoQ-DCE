import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# =============================================================================
# Loss function: Weighted sum of L1 loss and Mean Absolute Percentage Error (MAPE)
# =============================================================================
class MixedLoss(nn.Module):
    def __init__(self, alpha=0.998, eps=1e-6):
        """
        alpha: weight for L1 loss; (1-alpha) is the weight for MAPE.
        """
        super(MixedLoss, self).__init__()
        self.alpha = alpha
        self.eps = eps
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        l1_loss = self.l1(pred, target)
        # Compute MAPE; add epsilon to denominator to avoid division by zero.
        mape_loss = torch.mean(torch.abs((target - pred) / (target + self.eps)))
        return self.alpha * l1_loss + (1 - self.alpha) * mape_loss

class MPAE(nn.Module):
    def __init__(self, eps=1e-6):
        """
        alpha: weight for L1 loss; (1-alpha) is the weight for MAPE.
        """
        super(MPAE, self).__init__()
       
        self.eps = eps

    def forward(self, pred, target):
        # Compute MAPE; add epsilon to denominator to avoid division by zero.
        mape_loss = torch.mean(torch.abs((target - pred) / (target + self.eps)))
        return  mape_loss

def mpae(pred,target,eps=1e-6):
    return np.mean(abs((target-pred)/(target+eps)))


class MixedLossV2(nn.Module):
    def __init__(self, alpha=0.998, eps=1e-6, min_bound=None, max_bound=None, penalty_weight=1.0):
        """
        alpha: weight for L1 loss; (1-alpha) is the weight for MAPE.
        min_bound: tensor of shape (num_columns,) specifying lower bounds for each column.
        max_bound: tensor of shape (num_columns,) specifying upper bounds for each column.
        penalty_weight: weight for the boundary penalty term.
        """
        super(MixedLossV2, self).__init__()
        self.alpha = alpha
        self.eps = eps
        self.penalty_weight = penalty_weight
        self.l1 = nn.L1Loss()

        # Convert to tensors if min_bound and max_bound are provided as lists or scalars
        self.register_buffer("min_bound", torch.tensor(min_bound, dtype=torch.float32) if min_bound is not None else None)
        self.register_buffer("max_bound", torch.tensor(max_bound, dtype=torch.float32) if max_bound is not None else None)

    def forward(self, pred, target):
        # Ensure min_bound and max_bound are on the same device as pred
        device = pred.device
        min_bound = self.min_bound.to(device) if self.min_bound is not None else None
        max_bound = self.max_bound.to(device) if self.max_bound is not None else None

        # Compute primary loss
        l1_loss = self.l1(pred, target)
        mape_loss = torch.mean(torch.abs((target - pred) / (target + self.eps)))

        loss = self.alpha * l1_loss + (1 - self.alpha) * mape_loss

        # Apply penalty if pred is out of bounds
        if min_bound is not None:
            lower_violation = F.relu((min_bound - pred) / (min_bound + self.eps))  # Only penalize if pred < min_bound
        else:
            lower_violation = torch.zeros_like(pred)

        if max_bound is not None:
            upper_violation = F.relu((pred - max_bound) / (max_bound + self.eps))  # Only penalize if pred > max_bound
        else:
            upper_violation = torch.zeros_like(pred)

        boundary_penalty = torch.mean(lower_violation ** 2 + upper_violation ** 2)
        loss += self.penalty_weight * boundary_penalty

        return loss