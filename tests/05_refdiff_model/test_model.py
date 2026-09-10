import torch

from refdiff import ConditionalUNet1D, GaussianDiffusion, nmse
from refdiff.training import ExponentialMovingAverage, validate_noise


def test_conditional_unet_diffusion_forward_backward_and_ddim():
    torch.manual_seed(7)
    model = ConditionalUNet1D(base_channels=8, channel_multipliers=(1, 2, 4))
    diffusion = GaussianDiffusion(timesteps=12, schedule="cosine")
    target = torch.randn(2, 2, 64)
    condition = torch.randn_like(target)
    timestep = torch.tensor([0, 11])

    noisy, noise = diffusion.add_noise(target, timestep)
    prediction = model(noisy, condition, timestep)
    loss = (prediction - noise).square().mean()
    loss.backward()

    assert prediction.shape == target.shape
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())

    restored = diffusion.ddim_sample(model, condition, steps=4)
    assert restored.shape == condition.shape
    assert torch.isfinite(restored).all()


def test_nmse_is_strict_and_does_not_align_phase():
    reference = torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]])
    estimate = -reference
    values = nmse(estimate, reference)
    torch.testing.assert_close(values, torch.tensor([4.0, 4.0]))


def test_ema_starts_as_exact_model_copy():
    model = ConditionalUNet1D(base_channels=8, channel_multipliers=(1, 2))
    ema = ExponentialMovingAverage(model, decay=0.9)
    for source, averaged in zip(model.parameters(), ema.model.parameters(), strict=True):
        torch.testing.assert_close(source, averaged)
        assert not averaged.requires_grad


def test_noise_validation_reports_all_timestep_thirds():
    model = ConditionalUNet1D(base_channels=8, channel_multipliers=(1, 2))
    diffusion = GaussianDiffusion(timesteps=12, schedule="cosine")
    batch = {
        "target": torch.randn(4, 2, 32),
        "condition": torch.randn(4, 2, 32),
    }

    result = validate_noise(
        model,
        diffusion,
        [batch],
        torch.device("cpu"),
        seed=41,
    )

    assert all(
        torch.isfinite(torch.tensor(value))
        for value in (
            result.loss,
            result.low_t_loss,
            result.middle_t_loss,
            result.high_t_loss,
        )
    )
