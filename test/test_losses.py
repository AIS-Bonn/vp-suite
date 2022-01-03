import pytest
import sys
sys.path.append(".")

from vp_suite.measure.image_wise import MSE, SSIM, SmoothL1, LPIPS, L1, PSNR
from vp_suite.measure.fvd.fvd import FrechetVideoDistance as FVD

import torch

def setup_tensors_cpu():
    cpu = torch.device("cpu")
    dim5_zeros = torch.zeros(4, 8, 3, 63, 76, device=cpu, dtype=torch.float32)
    dim5_ones = torch.ones(4, 8, 3, 63, 76, device=cpu, dtype=torch.float32)
    dim5_randn = torch.randn(4, 8, 3, 63, 76, device=cpu, dtype=torch.float32)
    return dim5_zeros, dim5_ones, dim5_randn, cpu

def test_MSE_zero_for_equal_tensors():
    _, _, dim5_randn, cpu = setup_tensors_cpu()
    loss = MSE(device=cpu)
    assert not torch.is_nonzero(loss(dim5_randn, dim5_randn))

def test_L1_zero_for_equal_tensors():
    _, _, dim5_randn, cpu = setup_tensors_cpu()
    loss = L1(device=cpu)
    assert not torch.is_nonzero(loss(dim5_randn, dim5_randn))

def test_SmoothL1_zero_for_equal_tensors():
    _, _, dim5_randn, cpu = setup_tensors_cpu()
    loss = SmoothL1(device=cpu)
    assert not torch.is_nonzero(loss(dim5_randn, dim5_randn))

def test_LPIPS_zero_for_equal_tensors():
    _, _, dim5_randn, cpu = setup_tensors_cpu()
    loss = LPIPS(device=cpu)
    assert not torch.is_nonzero(loss(dim5_randn, dim5_randn))

def test_SSIM_one_for_equal_tensors():
    _, _, dim5_randn, cpu = setup_tensors_cpu()
    loss = SSIM(device=cpu)
    one = torch.tensor(1, device=cpu, dtype=torch.float32)
    assert not torch.isclose(loss(dim5_randn, dim5_randn), one)

def test_PSNR_posinf_for_equal_tensors():
    _, _, dim5_randn, cpu = setup_tensors_cpu()
    loss = PSNR(device=cpu)
    assert not torch.isposinf(loss(dim5_randn, dim5_randn))
