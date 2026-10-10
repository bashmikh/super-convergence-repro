"""CIFAR-10 held in device memory with batched on-device augmentation."""
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torchvision

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
PAD = 4


@dataclass
class Cifar:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor


def subset_indices(n: int, k: int, seed: int = 0) -> torch.Tensor:
    """Fixed random subset of size k; independent of the run seed."""
    if k > n:
        raise ValueError(f"subset {k} exceeds dataset size {n}")
    return torch.randperm(n, generator=torch.Generator().manual_seed(seed))[:k]


def _split(root, train: bool):
    ds = torchvision.datasets.CIFAR10(root, train=train, download=True)
    x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()
    return x, torch.tensor(ds.targets, dtype=torch.long)


def load_cifar10(root, device, train_subset: int | None = None) -> Cifar:
    """Load CIFAR-10 as uint8 NCHW tensors on `device`."""
    x_tr, y_tr = _split(root, True)
    x_te, y_te = _split(root, False)
    if train_subset is not None:
        idx = subset_indices(len(y_tr), train_subset)
        x_tr, y_tr = x_tr[idx], y_tr[idx]
    return Cifar(*(t.to(device) for t in (x_tr, y_tr, x_te, y_te)))


def fake_cifar(device, n_train: int = 2000, n_test: int = 500, seed: int = 0) -> Cifar:
    """Synthetic learnable data with CIFAR shapes: class-dependent intensity plus noise."""
    g = torch.Generator().manual_seed(seed)

    def make(n):
        y = torch.randint(0, 10, (n,), generator=g)
        x = y.view(-1, 1, 1, 1) * 25.0 + torch.randn(n, 3, 32, 32, generator=g) * 20 + 10
        return x.clamp(0, 255).to(torch.uint8), y

    x_tr, y_tr = make(n_train)
    x_te, y_te = make(n_test)
    return Cifar(*(t.to(device) for t in (x_tr, y_tr, x_te, y_te)))


def normalize(x: torch.Tensor) -> torch.Tensor:
    """uint8 -> float32 standardized with CIFAR-10 statistics."""
    mean = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x.float() / 255 - mean) / std


def crop_flip(x: torch.Tensor, i: torch.Tensor, j: torch.Tensor, flip: torch.Tensor) -> torch.Tensor:
    """Per-sample crop at offsets (i, j) of the reflect-padded batch, then optional h-flip."""
    n, c, h, w = x.shape
    dev = x.device
    p = F.pad(x, (PAD,) * 4, mode="reflect")
    rows = (i.to(dev)[:, None] + torch.arange(h, device=dev))[:, None, :, None]
    cols = (j.to(dev)[:, None] + torch.arange(w, device=dev))[:, None, None, :]
    out = p[torch.arange(n, device=dev)[:, None, None, None],
            torch.arange(c, device=dev)[None, :, None, None], rows, cols]
    return torch.where(flip.to(dev)[:, None, None, None], out.flip(3), out)


def augment(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Random 32x32 crop with 4-pixel reflect padding and random horizontal flip."""
    n, dev = x.shape[0], x.device
    i = torch.randint(0, 2 * PAD + 1, (n,), device=dev, generator=gen)
    j = torch.randint(0, 2 * PAD + 1, (n,), device=dev, generator=gen)
    flip = torch.rand(n, device=dev, generator=gen) < 0.5
    return crop_flip(x, i, j, flip)
