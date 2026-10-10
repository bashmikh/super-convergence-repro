import os

import pytest
import torch

from convergence_repro import data
from convergence_repro.data import (PAD, augment, crop_flip, fake_cifar, normalize,
                                    subset_indices)


def test_normalize_shape_dtype():
    x = torch.randint(0, 256, (4, 3, 32, 32), dtype=torch.uint8)
    y = normalize(x)
    assert y.dtype == torch.float32 and y.shape == x.shape
    assert x.dtype == torch.uint8


def test_crop_flip_identity_and_shift():
    x = torch.randn(3, 3, 32, 32)
    n = x.shape[0]
    center = torch.full((n,), PAD)
    no = torch.zeros(n, dtype=torch.bool)
    assert torch.equal(crop_flip(x, center, center, no), x)
    assert torch.equal(crop_flip(x, center, center, ~no), x.flip(3))
    zero = torch.zeros(n, dtype=torch.long)
    shifted = crop_flip(x, zero, zero, no)
    assert torch.equal(shifted[..., PAD:, PAD:], x[..., :32 - PAD, :32 - PAD])


def test_augment_shape_and_determinism():
    x = torch.randn(8, 3, 32, 32)
    a = augment(x, torch.Generator().manual_seed(0))
    b = augment(x, torch.Generator().manual_seed(0))
    assert a.shape == x.shape and torch.equal(a, b)


def test_subset_indices():
    a, b = subset_indices(50_000, 10_000), subset_indices(50_000, 10_000)
    assert torch.equal(a, b) and len(a.unique()) == 10_000
    with pytest.raises(ValueError):
        subset_indices(10, 11)


def test_fake_cifar():
    d = fake_cifar("cpu", n_train=200, n_test=50)
    assert d.x_train.dtype == torch.uint8 and d.x_train.shape == (200, 3, 32, 32)
    assert d.y_train.dtype == torch.long and d.y_test.shape == (50,)


class FakeCifar:
    """Stands in for torchvision CIFAR10: 'downloads' only when the archive is absent."""

    downloads = 0
    tgz_md5 = "fake"

    def __init__(self, root, train, download):
        from pathlib import Path

        archive = Path(root) / data.ARCHIVE
        if (archive.read_bytes() if archive.exists() else b"") != b"good":
            FakeCifar.downloads += 1
            archive.write_bytes(b"good")


@pytest.fixture
def fake_tv(monkeypatch):
    FakeCifar.downloads = 0
    monkeypatch.setattr(data.torchvision.datasets, "CIFAR10", FakeCifar)
    monkeypatch.setattr(data, "check_integrity",
                        lambda path, md5=None: os.path.exists(path) and open(path, "rb").read() == b"good")


def test_fetch_fills_empty_cache(tmp_path, fake_tv):
    data.fetch_cifar10(tmp_path / "data", tmp_path / "cache")
    assert FakeCifar.downloads == 1
    assert (tmp_path / "cache" / data.ARCHIVE).read_bytes() == b"good"


def test_fetch_uses_cache(tmp_path, fake_tv):
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / data.ARCHIVE).write_bytes(b"good")
    data.fetch_cifar10(tmp_path / "data", tmp_path / "cache")
    assert FakeCifar.downloads == 0


def test_fetch_replaces_corrupt_cache(tmp_path, fake_tv):
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / data.ARCHIVE).write_bytes(b"partial")
    data.fetch_cifar10(tmp_path / "data", tmp_path / "cache")
    assert FakeCifar.downloads == 1
    assert (tmp_path / "cache" / data.ARCHIVE).read_bytes() == b"good"
