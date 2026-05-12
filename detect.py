import argparse
import os

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.paths = sorted([
            os.path.join(dirpath, f)
            for dirpath, _, filenames in os.walk(root)
            for f in filenames
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        ])
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def gaussian_blur(tensor, sigma):
    out = torch.empty_like(tensor)
    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            out[i, j] = torch.tensor(
                gaussian_filter(tensor[i, j].cpu().numpy(), sigma=sigma)
            )
    return out


def gaussian_noise(tensor, mean, std):
    return tensor + torch.randn(tensor.size()) * std + mean


def salt_and_pepper_noise(tensor, salt_prob, pepper_prob):
    noise = torch.rand(tensor.size(), device=tensor.device)
    out = tensor.clone()
    out[noise < salt_prob] = 1.0
    out[noise > (1 - pepper_prob)] = -1.0
    return out


def rotate(tensor, degrees):
    t = transforms.RandomRotation(degrees)
    return torch.stack([t(tensor[i]) for i in range(tensor.shape[0])])


def build_transform(args):
    if args.transform == "gaussian_noise":
        return lambda x: gaussian_noise(x, args.mean, args.std)
    if args.transform == "gaussian_blur":
        return lambda x: gaussian_blur(x, args.sigma)
    if args.transform == "salt_and_pepper":
        return lambda x: salt_and_pepper_noise(x, args.salt_prob, args.pepper_prob)
    if args.transform == "rotate":
        return lambda x: rotate(x, tuple(args.degrees))


def extract_features(tensor, model):
    with torch.no_grad():
        fx = model(tensor.half().cuda())
    return fx / fx.norm(dim=-1, keepdim=True)


def pairwise_inner_products(features):
    results = []
    for i in range(features.size(0)):
        others = torch.cat([features[:i], features[i + 1:]], dim=0)
        results.append(torch.matmul(others, features[i].unsqueeze(1)))
    return results


def energy_scores(x, tx, temperature, model):
    fx_ip = pairwise_inner_products(extract_features(x, model))
    ftx_ip = pairwise_inner_products(extract_features(tx, model))
    return [
        torch.norm(torch.exp(v1 / temperature) - torch.exp(v2 / temperature), p=1).cpu().item()
        for v1, v2 in zip(fx_ip, ftx_ip)
    ]


def collect_scores(path, label, model, transform_fn, batch_size, temperature, max_batches=20):
    loader = DataLoader(ImageDataset(path, PREPROCESS), batch_size=batch_size, shuffle=True)
    scores = []
    for i, x in enumerate(tqdm(loader, desc=label, unit="batch")):
        scores += energy_scores(x, transform_fn(x), temperature, model)
        if max_batches > 0 and i + 1 >= max_batches:
            break
    return scores


def compute_metrics(real_scores, fake_scores):
    scores = np.concatenate([fake_scores, real_scores])
    labels = np.array([1] * len(fake_scores) + [0] * len(real_scores), dtype=np.int32)
    auroc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    return auroc, ap


def find_best_threshold(real_scores, fake_scores):
    d0, d1 = np.array(real_scores), np.array(fake_scores)
    all_sorted = np.sort(np.concatenate([d0, d1]))
    thresholds = np.concatenate([[-np.inf], (all_sorted[:-1] + all_sorted[1:]) / 2, [np.inf]])
    d0s, d1s = np.sort(d0), np.sort(d1)
    total = len(d0) + len(d1)
    max_correct = -1
    best_ths = []

    for th in thresholds:
        c0 = np.searchsorted(d0s, th, "right") + len(d1) - np.searchsorted(d1s, th, "right")
        c1 = len(d0) - np.searchsorted(d0s, th, "right") + np.searchsorted(d1s, th, "right")
        cur = max(c0, c1)
        if cur > max_correct:
            max_correct, best_ths = cur, [th]
        elif cur == max_correct:
            best_ths.append(th)

    return float(np.median(best_ths)), max_correct / total


def parse_args():
    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument("--real", required=True, metavar="DIR",
                        help="Directory of real images.")
    parser.add_argument("--fake", required=True, metavar="DIR",
                        help="Directory of generated images.")

    # Model
    parser.add_argument("--model", default="dinov2_vitl14",
                        choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"],
                        help="DINOv2 model variant (default: dinov2_vitl14).")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size (default: 128).")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Temperature (default: 0.6).")

    # Random transformation
    parser.add_argument("--transform", default="gaussian_noise",
                        choices=["gaussian_noise", "gaussian_blur", "salt_and_pepper", "rotate"],
                        help="Random transformation method (default: gaussian_noise).")
    parser.add_argument("--mean", type=float, default=0.0,
                        help="[gaussian_noise] Noise mean (default: 0.0).")
    parser.add_argument("--std", type=float, default=0.04,
                        help="[gaussian_noise] Noise std (default: 0.04).")
    parser.add_argument("--sigma", type=float, default=0.7,
                        help="[gaussian_blur] Blur sigma (default: 0.7).")
    parser.add_argument("--salt-prob", type=float, default=1e-4,
                        help="[salt_and_pepper] Salt probability (default: 1e-4).")
    parser.add_argument("--pepper-prob", type=float, default=1e-4,
                        help="[salt_and_pepper] Pepper probability (default: 1e-4).")
    parser.add_argument("--degrees", type=float, nargs=2, default=[-10, 10],
                        metavar=("MIN", "MAX"),
                        help="[rotate] Rotation range in degrees (default: -10 10).")
    parser.add_argument("--max-batches", type=int, default=20,
                        help="Maximum number of batches to evaluate (default: 20).")

    return parser.parse_args()


def main():
    args = parse_args()
    transform_fn = build_transform(args)
    model = torch.hub.load("facebookresearch/dinov2", args.model)
    model = model.half().cuda().eval()

    real_scores = collect_scores(args.real, "real", model, transform_fn, args.batch_size, args.temperature, args.max_batches)
    fake_scores = collect_scores(args.fake, "fake", model, transform_fn, args.batch_size, args.temperature, args.max_batches)

    auroc, ap = compute_metrics(real_scores, fake_scores)
    _, acc = find_best_threshold(real_scores, fake_scores)

    print(f"AUROC: {auroc:.4f}")
    print(f"AP:    {ap:.4f}")
    print(f"ACC:   {acc:.4f}")


if __name__ == "__main__":
    main()
