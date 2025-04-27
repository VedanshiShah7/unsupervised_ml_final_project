import torch
import numpy as np

import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode


def get_target_label_idx(labels, targets):
    """
    Get the indices of labels that are included in targets.
    :param labels: array of labels
    :param targets: list/tuple of target labels
    :return: list with indices of target labels
    """
    return np.argwhere(np.isin(labels, targets)).flatten().tolist()

def custom_transforms_cifar(min_val, range_val):
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),

        
        transforms.ToTensor(),
        transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),  # <- your original GCN
        transforms.Normalize([min_val] * 3, [range_val] * 3)
    ])
    return transform

def custom_transforms_mnist(min_val, range_val):
    transform = transforms.Compose([

        transforms.ToTensor(),
        transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),  # <- your original GCN
        transforms.Normalize([min_val] * 3, [range_val] * 3)
    ])
    return transform

def global_contrast_normalization(x: torch.tensor, scale='l2'):
    """
    Apply global contrast normalization to tensor, i.e. subtract mean across features (pixels) and normalize by scale,
    which is either the standard deviation, L1- or L2-norm across features (pixels).
    Note this is a *per sample* normalization globally across features (and not across the dataset).
    """

    assert scale in ('l1', 'l2')

    n_features = int(np.prod(x.shape))

    mean = torch.mean(x)  # mean over all features (pixels) per sample
    x -= mean

    if scale == 'l1':
        x_scale = torch.mean(torch.abs(x))

    if scale == 'l2':
        x_scale = torch.sqrt(torch.sum(x ** 2)) / n_features

    x /= x_scale

    return x
