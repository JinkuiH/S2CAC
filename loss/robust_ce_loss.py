import torch
from torch import nn, Tensor
import numpy as np
import torch.nn.functional as F

        
# class RobustCrossEntropyLossV2(nn.CrossEntropyLoss):
#     def __init__(self, **kwargs):
#         """
#         RobustCrossEntropyLoss with sample weighting.
#         """
#         super(RobustCrossEntropyLossV2, self).__init__(**kwargs)

#     def forward(self, pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor):
#         """
#         pred: predicted logits, shape (n, c, x, y, z)
#         target: ground truth, shape (n, 1, x, y, z)
#         weight: sample-wise weights, shape (n,)
#         """
#         if target.ndim == pred.ndim:
#             target = target[:, 0]  # Remove channel dimension

#         # Flatten spatial dimensions for weighted loss
#         pred_flat = pred.view(pred.shape[0], pred.shape[1], -1)  # Shape (n, c, spatial_size)
#         target_flat = target.view(target.shape[0], -1)  # Shape (n, spatial_size)
#         weight_flat = weight.view(-1, 1)  # Shape (n, 1)

#         # Compute cross-entropy loss with per-sample weight
#         log_prob = F.log_softmax(pred_flat, dim=1)  # Log probabilities
#         target_one_hot = F.one_hot(target_flat.long(), num_classes=pred.shape[1])  # Convert target to one-hot
#         target_one_hot = target_one_hot.permute(0, 2, 1).float()  # Shape (n, c, spatial_size)

#         loss = -torch.sum(weight_flat * target_one_hot * log_prob, dim=(1, 2))  # Weighted loss per sample
#         return loss.mean()  # Average across batch
class RobustCrossEntropyLossWeight(nn.CrossEntropyLoss):
    """
    CrossEntropyLoss with sample weighting.
    """
    def forward(self, input: Tensor, target: Tensor, weight: Tensor) -> Tensor:
        """
        input: logits, shape (n, c, ...)
        target: ground truth, shape (n, ...)
        weight: sample weights, shape (n,)
        """
        # Ensure target has the correct shape
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]

        # Compute the per-sample loss
        per_sample_loss = super().forward(input, target.long())
        
        # Apply the weights to the loss
        weighted_loss = (per_sample_loss * weight).sum()/weight.sum()
        
        # Return the weighted average loss
        return weighted_loss
    
class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension

    input must be logits, not probabilities!
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())


class TopKLoss(RobustCrossEntropyLoss):
    """
    input must be logits, not probabilities!
    """
    def __init__(self, weight=None, ignore_index: int = -100, k: float = 10, label_smoothing: float = 0):
        self.k = k
        super(TopKLoss, self).__init__(weight, False, ignore_index, reduce=False, label_smoothing=label_smoothing)

    def forward(self, inp, target):
        target = target[:, 0].long()
        res = super(TopKLoss, self).forward(inp, target)
        num_voxels = np.prod(res.shape, dtype=np.int64)
        res, _ = torch.topk(res.view((-1, )), int(num_voxels * self.k / 100), sorted=False)
        return res.mean()
