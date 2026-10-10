import pytest
import torch

from convergence_repro.models import resnet_cifar


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def test_resnet20_shape_and_size():
    m = resnet_cifar(20)
    assert m(torch.randn(4, 3, 32, 32)).shape == (4, 10)
    assert 0.26e6 < n_params(m) < 0.28e6


def test_resnet56_builds():
    m = resnet_cifar(56)
    assert m(torch.randn(2, 3, 32, 32)).shape == (2, 10)
    assert 0.84e6 < n_params(m) < 0.87e6


def test_bn_momentum_applied():
    m = resnet_cifar(8, bn_momentum=0.05)
    bns = [x for x in m.modules() if isinstance(x, torch.nn.BatchNorm2d)]
    assert bns and all(b.momentum == 0.05 for b in bns)


def test_invalid_depth():
    with pytest.raises(ValueError):
        resnet_cifar(21)


def test_channels_last_forward():
    m = resnet_cifar(8).to(memory_format=torch.channels_last)
    x = torch.randn(2, 3, 32, 32).contiguous(memory_format=torch.channels_last)
    assert m(x).shape == (2, 10)
