import nibabel as nib
import numpy as np
from monai.data import MetaTensor
import torch
import torch.nn as nn
from monai.transforms import (
    EnsureChannelFirst,
    EnsureChannelFirstd,
    EnsureChannelFirstD,
    Compose,
    ScaleIntensityd,
    ResizeD,
    RandFlipd,
    RandRotate90d,
    RandAdjustContrastd,
    RandShiftIntensityd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    MapTransform,
    ToTensord
)
import cv2


def get_augmentation_transforms():
    return Compose([
        EnsureChannelFirstd(keys=["image", "segmentation"], channel_dim="no_channel", allow_missing_keys=True),
        RandAdjustContrastd(keys=["image"], prob=0.4, gamma=(0.8, 1.2)),
        RandShiftIntensityd(keys=["image"], prob=0.4, offsets=(-0.1, 0.1)),
        RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.05),
        # RandGaussianSmoothd(keys=["image"], prob=0.5, sigma_x=(0.5, 2.0), sigma_y=(0.5, 2.0), sigma_z=(0.5, 2.0)),

        RandFlipd(keys=["image", "segmentation"], prob=0.5, spatial_axis=0),
        RandRotate90d(keys=["image", "segmentation"], prob=0.3, spatial_axes=(0, 1)),
    ])

def get_non_aug_transforms():
    return Compose([
        EnsureChannelFirstd(keys=["image", "segmentation"], channel_dim="no_channel", allow_missing_keys=True),
    ])


class SegmentationBasedSharpenBlur(MapTransform):
    def __init__(self, keys, seg_key, alpha=1.0, sigma=1.0):
        super().__init__(keys)
        self.seg_key = seg_key
        self.alpha = alpha
        self.sigma = sigma

    def _laplacian_sharpen(self, image):
        orig_dtype = image.dtype
        image_64 = image.astype(np.float64)
        laplacian = cv2.Laplacian(image_64, cv2.CV_64F)
        sharpened = image_64 - self.alpha * laplacian

        if np.issubdtype(orig_dtype, np.uint8):
            sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        elif np.issubdtype(orig_dtype, np.floating):
            sharpened = np.clip(sharpened, 0.0, 1.0).astype(orig_dtype)
        else:
            sharpened = sharpened.astype(orig_dtype)
        return sharpened

    def _gaussian_blur(self, image):
        return cv2.GaussianBlur(image, (0, 0), self.sigma)

    def __call__(self, data):
        d = dict(data)
        if self.seg_key not in d:
            raise KeyError(f"segmentation key '{self.seg_key}' not in dict")
        seg = d[self.seg_key]
        if not isinstance(seg, np.ndarray):
            seg = seg.cpu().numpy() if hasattr(seg, "cpu") else np.array(seg)

        seg_mask = (seg > 0).astype(np.float32)
        non_seg_mask = (seg <= 0).astype(np.float32)

        for key in self.keys:
            image = d[key]
            if not isinstance(image, np.ndarray):
                image = image.cpu().numpy() if hasattr(image, "cpu") else np.array(image)

            if image.ndim == 2:
                sharpened = self._laplacian_sharpen(image)
                blurred = self._gaussian_blur(image)
                combined = seg_mask * sharpened + (1 - seg_mask) * blurred
                d[key] = combined.astype(image.dtype)

            elif image.ndim == 3:
                channels = []
                for c in range(image.shape[0]):
                    channel = image[c]
                    if seg_mask.ndim == 2:
                        mask_channel = seg_mask
                    elif seg_mask.ndim == 3:
                        if seg_mask.shape[0] == image.shape[0]:
                            mask_channel = seg_mask[c]
                        else:
                            raise ValueError(f"segmentation mask shape {seg_mask.shape} do not match image-channel")
                    else:
                        raise ValueError(f"non-supported segmentation mask dim: {seg_mask.ndim}")

                    sharpened = self._laplacian_sharpen(channel)
                    blurred = self._gaussian_blur(channel)
                    combined = mask_channel * sharpened + (1 - mask_channel) * blurred
                    channels.append(combined)
                d[key] = np.stack(channels, axis=0).astype(image.dtype)

            elif image.ndim == 4:
                channels = []
                for c in range(image.shape[0]):
                    channel = image[c]  # shape: (H, W, D)
                    if seg_mask.ndim == 3:
                        mask_channel = seg_mask
                    elif seg_mask.ndim == 4:
                        if seg_mask.shape[0] == image.shape[0]:
                            mask_channel = seg_mask[c]
                            non_mask_channel = non_seg_mask[c]
                        else:
                            raise ValueError(f"segmentation mask shape {seg_mask.shape} do not match image-channel")
                    else:
                        raise ValueError(f"non-supported segmentation mask dim: {seg_mask.ndim}")

                    processed_slices = []
                    for d_idx in range(channel.shape[-1]):
                        slice_img = channel[..., d_idx]   # (H, W)
                        slice_mask = mask_channel[..., d_idx]  # (H, W)
                        slice_non_mask = non_mask_channel[..., d_idx]  # (H, W)
                        sharpened = self._laplacian_sharpen(slice_img)
                        blurred = self._gaussian_blur(slice_img)
                        combined = slice_non_mask * blurred + slice_mask * sharpened
                        processed_slices.append(combined)
                    channel_processed = np.stack(processed_slices, axis=-1)
                    channels.append(channel_processed)
                d[key] = np.stack(channels, axis=0).astype(image.dtype)

            else:
                raise ValueError(f"non-supported segmentation img dim: {image.ndim}")

        return d