# -*- coding:utf-8 -*-
"""
作者：李钰钦
日期：2025年10月27日
"""
import torch
import yaml
from models.layers import CBAM
from models import Record


def test_cbam_module():
    """Test CBAM module independently"""
    batch_size = 2
    channels = 64
    height, width = 32, 32

    # Test CBAM module
    cbam = CBAM(in_channels=channels, reduction_ratio=16, kernel_size=7)
    x = torch.randn(batch_size, channels, height, width)

    output = cbam(x)

    assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"
    print(f"✓ CBAM module test passed! Input: {x.shape} -> Output: {output.shape}")

    # Test with different parameters
    cbam_small = CBAM(in_channels=channels, reduction_ratio=8, kernel_size=3)
    output_small = cbam_small(x)
    assert output_small.shape == x.shape
    print(f"✓ CBAM (reduction=8, kernel=3) test passed!")


def test_record_with_cbam():
    """Test RECORD model with CBAM"""
    # Load config
    with open('configs/config_record_cruw.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Load backbone config
    with open(config['model_cfg']['backbone_pth'], 'r') as f:
        backbone_cfg = yaml.safe_load(f)

    # Create model without CBAM
    model_no_cbam = Record(
        config=backbone_cfg,
        in_channels=8,
        n_class=3,
        use_cbam=False
    )

    # Create model with CBAM
    model_with_cbam = Record(
        config=backbone_cfg,
        in_channels=8,
        n_class=3,
        use_cbam=True,
        cbam_reduction=16,
        cbam_kernel_size=7
    )

    # Test forward pass
    batch_size = 2
    n_frames = 12
    x = torch.randn(batch_size, 8, n_frames, 128, 128)

    # Without CBAM
    output_no_cbam = model_no_cbam(x)
    print(f"✓ Model without CBAM: {x.shape} -> {output_no_cbam.shape}")

    # With CBAM
    output_with_cbam = model_with_cbam(x)
    print(f"✓ Model with CBAM: {x.shape} -> {output_with_cbam.shape}")

    # Count parameters
    params_no_cbam = sum(p.numel() for p in model_no_cbam.parameters())
    params_with_cbam = sum(p.numel() for p in model_with_cbam.parameters())

    print(f"\nParameter comparison:")
    print(f"  Without CBAM: {params_no_cbam:,} parameters")
    print(f"  With CBAM: {params_with_cbam:,} parameters")
    print(
        f"  Increase: {params_with_cbam - params_no_cbam:,} parameters ({(params_with_cbam / params_no_cbam - 1) * 100:.2f}%)")


if __name__ == '__main__':
    print("Testing CBAM module...")
    test_cbam_module()
    print("\n" + "=" * 50 + "\n")
    print("Testing RECORD model with CBAM...")
    test_record_with_cbam()
