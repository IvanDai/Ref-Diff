import torch

from rfstyle.models import ConditionalUNet1D
from rfstyle.training import GaussianDiffusion


def test_model_and_diffusion_shapes():
    model = ConditionalUNet1D(base_channels=8, channel_mults=(1, 2, 4))
    target = torch.randn(2, 2, 128)
    condition = torch.randn_like(target)
    diffusion = GaussianDiffusion(10, 1e-4, 0.02, "cpu")
    t = torch.tensor([1, 5])
    noisy, noise = diffusion.q_sample(target, t)
    prediction = model(noisy, condition, t)
    assert prediction.shape == noise.shape

