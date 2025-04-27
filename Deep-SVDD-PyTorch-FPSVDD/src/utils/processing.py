import torch
import numpy as np

def batch_mixup(x, y, alpha=1.0, num_classes=10):
    """Applies Mixup on an entire batch."""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)

    # Shuffle the batch
    index = torch.randperm(batch_size)
    x2 = x[index]
    y2 = y[index]

    # Mix images
    mixed_x = lam * x + (1 - lam) * x2

    # If labels are not one-hot yet, convert
    if y.ndim == 1:
        y_onehot = torch.zeros(batch_size, num_classes, device=x.device)
        y_onehot.scatter_(1, y.unsqueeze(1), 1)
    else:
        y_onehot = y

    if y2.ndim == 1:
        y2_onehot = torch.zeros(batch_size, num_classes, device=x.device)
        y2_onehot.scatter_(1, y2.unsqueeze(1), 1)
    else:
        y2_onehot = y2

    # Mix labels
    mixed_y = lam * y_onehot + (1 - lam) * y2_onehot

    return mixed_x, mixed_y

