# Data & preprocessing

This document covers how raw NIfTI volumes become the 2-channel tensors fed to the
models. See [architecture.md](architecture.md) for how these pieces are wired in.

## Inputs

Each sample combines two 3D NIfTI files:

- **MRI image** (`image_path`)
- **Tumour segmentation mask** (`seg_v<version>_path`)

They are stacked along the channel dimension into a `(2, H, W, D)` tensor —
**channel 0 = image, channel 1 = segmentation**. The classification label is a
patient-level clinical target (e.g. Huvos chemotherapy response), binary `{0, 1}`.

## Loading — `src/data/load_datapath.py`

`load_os_by_modality_version(modality, version, return_subjects=False)`:

1. Reads `preprocessing/<modality>_df.csv` (one row per image).
2. Rewrites stored file paths to the current cluster mount points
   (`/exports/...` → `/projects/...`).
3. Keeps only rows with `included == 'yes'`.
4. Selects the segmentation column `seg_v<version>_path` (raises if the version is
   absent, listing available ones).
5. Reads `clinical_features/clinical_features_factorized.csv` and builds a
   `Patient → label` map (last column = target).
6. Matches each image's subject to a label (a subject-ID normalisation
   `OS_0 → OS_00` is applied), skipping unmatched subjects with a warning.

Returns aligned lists: `image_files, segmentation_files, labels[, subject_ids]`.
It also prints image/subject counts and the label distribution.

> These CSV locations are hard-coded absolute paths; change them for a new
> environment.

## Patient splits

Outer-fold splits are **not** computed at runtime — they are read from a CSV passed
via `--split_file`. Each outer fold `i` is a pair of columns `<i>_train` /
`<i>_test`, each a list of patient IDs. `load_predefined_splits` parses these into a
list of `{'train': [...], 'test': [...]}` dicts, and patients absent from the loaded
data are dropped. This guarantees the outer test set is defined at the **patient**
level (no image-level leakage).

Inner CV then re-splits the outer-train patients with `StratifiedKFold` — again at
the patient level (see [cv_framework.py](../src/cross_validation/cv_framework.py)).

## Dataset — `src/data/dataset.py`

`OsteosarcomaDataset` performs per-sample loading, preprocessing, caching, and
augmentation.

### Preprocessing pipeline (`_preprocess_pipeline`)

Applied once per original sample, then cached (`cache_data=True`):

1. **Load NIfTI** (`_load_nifti`) — image as float32, segmentation as int32;
   spacing derived from the affine. If image/seg affines differ, the segmentation
   is resampled onto the image grid (nearest-neighbour).
2. **Swap axes** (`_swap_axes_to_standard`) — transpose so the **shortest axis is
   last** (a consistent `(H, W, D)` orientation), adjusting spacing accordingly.
3. **Resample to target spacing** (`_resample_to_spacing`) — `scipy.ndimage.zoom`,
   linear for images, nearest for segmentations. Default target spacing
   `(1.5, 1.5, 3.0)` mm.
4. **Crop / pad to target size** (`_crop_or_pad`) — default `(192, 192, 64)`.
   - Crop strategy `foreground` centres the crop on the segmentation bounding box
     (falls back to centre crop when no foreground is present); `center` always
     centre-crops.
   - Padding uses a background value estimated from image corners, and a
     `valid_mask` records which voxels are real vs padded.

An intensity-normalisation helper (`_normalize_intensity`, z-score on valid voxels
with percentile clipping) exists in the dataset, but in the current loader
normalization is done in the transform pipeline instead (see below).

### Augmentation & caching

- The base preprocessed volume is cached per original sample; augmentations are
  applied on top, so caching does not leak augmentation state.
- `num_augmentations > 1` makes `__len__` report `n_samples × num_augmentations`;
  `__getitem__` maps a flat index to `(original_idx, augmentation_idx)` and seeds
  the transform deterministically per augmentation so repeated epochs are
  reproducible. Augmentation only applies when `is_train=True`.

## Transforms — `src/data/transform.py`

MONAI dictionary transforms operating on `{"image", "segmentation"}`:

- **`get_augmentation_transforms()`** (training):
  - Intensity: `RandAdjustContrast`, `RandShiftIntensity`, `RandGaussianNoise`,
    `RandGaussianSmooth` (image only).
  - Spatial: `RandFlip` on all three axes, `RandRotate90` in-plane (image + seg
    together).
  - `NormalizeIntensity` (per-channel, per-sample z-score).
- **`get_non_aug_transforms()`** (validation/test): channel-first + the same
  `NormalizeIntensity` only.
- **`SegmentationBasedSharpenBlur`** — an optional custom `MapTransform` that
  sharpens inside the tumour mask and blurs outside (2D/3D/4D supported). Not part
  of the default pipeline.

## Loaders — `src/cross_validation/cv_framework.py`

`create_data_loaders` builds three datasets from the split tuples:

| Split | Transform | `num_augmentations` | `shuffle` |
|---|---|---|---|
| train | augmentation | `hyperparams['num_augmentations']` | ✓ |
| val   | non-aug | 1 | ✗ |
| test  | non-aug | 1 | ✗ |

All use `batch_size` from `hyperparams`, `num_workers=10`, and
`pin_memory=(device is CUDA)`. Dataset parameters (`target_spacing`, `target_size`,
`normalize`, `crop_strategy`) are read from `hyperparams` with the defaults noted
above.

## Pseudo-label variant — `src/data/pdataset.py`

`OsteosarcomaDatasetWithPseudoLabels` derives labels from segmentation-based
measurements (e.g. tumour volume, max/mean diameter) thresholded by a chosen method
(median/mean/percentile/manual). The hooks for it are present but commented out in
`create_data_loaders`; `analysis/check_pseudo_balance.py` inspects the resulting
class balance.
</content>
