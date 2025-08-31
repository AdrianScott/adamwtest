import torch

from custom_optimizer import CustomAdamW


def test_cumulative_beta_products_with_constant_betas():
    # With dynamic_smoothing=False, betas are constant each step
    w = torch.nn.Parameter(torch.ones(1, requires_grad=True))
    opt = CustomAdamW([w], lr=1e-2, betas=(0.9, 0.99), dynamic_smoothing=False)

    for step in range(1, 6):
        opt.zero_grad()
        w.sum().backward()
        opt.step()
        st = opt.state[w]
        beta1_prod = st['beta1_prod'].item()
        beta2_prod = st['beta2_prod'].item()
        assert abs(beta1_prod - (0.9 ** step)) < 1e-7
        assert abs(beta2_prod - (0.99 ** step)) < 1e-7

