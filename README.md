# DEnD-Detection

This is the official repo of the paper "Towards Generalizable Detector for Generated Image".

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

```bash
python detect.py --real path/to/real/images --fake path/to/fake/images
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--real` | *(required)* | Directory of real images |
| `--fake` | *(required)* | Directory of generated images |
| `--model` | `dinov2_vitl14` | DINOv2 variant: `dinov2_vits14` / `dinov2_vitb14` / `dinov2_vitl14` / `dinov2_vitg14` |
| `--batch-size` | `128` | Batch size |
| `--temperature` | `0.6` | Temperature |
| `--transform` | `gaussian_noise` | Random transformation: `gaussian_noise` / `gaussian_blur` / `salt_and_pepper` / `rotate` |
| `--max-batches` | `20` | Max batches to evaluate per class (`-1` for full dataset) |
