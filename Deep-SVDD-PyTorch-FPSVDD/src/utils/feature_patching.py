import torch.nn.functional as F
import torch

def patch_and_aggregate_features(feature_map, patch_size=3):
    B, C, H, W = feature_map.shape
    patches = F.unfold(feature_map, kernel_size=patch_size, padding=patch_size // 2)
    patches = patches.permute(0, 2, 1)  # (B, num_patches, C * patch^2)
    return patches


def extract_multi_scale_patches(features_dict, patch_size=3):
    """
    Concatenate unfolded feature patches across CNN layers.
    Returns: (B, N_patches, D_total)
    """
    patch_list = []
    ref_shape = max((f.shape[2], f.shape[3]) for f in features_dict.values())

    for feat in features_dict.values():
        if feat.shape[2:] != ref_shape:
            feat = F.interpolate(feat, size=ref_shape, mode='bilinear', align_corners=False)
        patches = F.unfold(feat, kernel_size=patch_size, padding=patch_size // 2)
        patches = patches.permute(0, 2, 1)  # (B, N_patches, D)
        patch_list.append(patches)

    return torch.cat(patch_list, dim=-1)
