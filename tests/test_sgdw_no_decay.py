import torch
import torch.nn as nn

from sweep_train import SGDW, param_groups_weight_decay


def test_sgdw_respects_no_decay_group():
    torch.manual_seed(0)
    m = nn.Linear(2, 2, bias=True)
    # Build param groups with no_decay for bias
    groups = param_groups_weight_decay(m, weight_decay=0.1, exclude_bias_norm=True)
    opt = SGDW(groups, lr=0.01, momentum=0.0, weight_decay=0.1)

    # Zero gradients but ensure grad tensors exist so WD applies
    for p in m.parameters():
        p.grad = torch.zeros_like(p)

    w_before = m.weight.detach().clone()
    b_before = m.bias.detach().clone()

    opt.step()

    # Weight should be decayed, bias should not
    decay_factor = 1 - 0.01 * 0.1
    assert torch.allclose(m.weight, w_before * decay_factor, atol=0, rtol=0), "weight should be decayed"
    assert torch.allclose(m.bias, b_before, atol=0, rtol=0), "bias should not be decayed"

