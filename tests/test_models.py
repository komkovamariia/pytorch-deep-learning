import pytest
import torch

from deep_projects.models import (
    DenseAutoencoder,
    Diffusion,
    Discriminator,
    ELBOLoss,
    Generator,
    GridGCN,
    NoisePredictor,
    RecurrentForecaster,
    SignalCNN,
    SignalGRU,
    SpikingMLP,
    SurrogateSpike,
    VariationalAutoencoder,
    grid_adjacency,
)


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(1031)
    torch.set_num_threads(2)


def test_signal_shapes():
    signals = torch.randn(2, 140)
    assert DenseAutoencoder()(signals).shape == signals.shape
    assert SignalCNN()(signals).shape == (2, 5)
    assert SignalGRU()(signals).shape == (2, 5)
    for cell in ["rnn", "gru", "lstm"]:
        assert RecurrentForecaster(cell)(torch.rand(2, 24, 1)).shape == (2, 1)


def test_grid_normalization_matches_degrees():
    adjacency = grid_adjacency(8)
    assert adjacency.shape == (64, 64)
    assert torch.equal(adjacency, adjacency.T)
    assert torch.isclose(adjacency[0, 0], torch.tensor(1 / 3))
    assert torch.isclose(adjacency[0, 1], torch.tensor(1 / (3 * 4) ** 0.5))
    assert adjacency[0, 2] == 0
    assert GridGCN(adjacency)(torch.rand(2, 1, 8, 8)).shape == (2, 10)


def test_surrogate_backward_is_finite_and_local():
    voltages = torch.tensor([-10.0, -0.1, 0.0, 0.1, 10.0], requires_grad=True)
    spikes = SurrogateSpike.apply(voltages)
    assert torch.equal(spikes, torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0]))
    spikes.sum().backward()
    assert torch.isfinite(voltages.grad).all()
    assert voltages.grad[2] > voltages.grad[0]
    model = SpikingMLP()
    model(torch.rand(3, 1, 8, 8)).sum().backward()
    assert model.input_layer.weight.grad.abs().sum() > 0


def test_vae_kl_and_deterministic_evaluation():
    images = torch.rand(3, 1, 8, 8)
    model = VariationalAutoencoder()
    model.eval()
    output = model(images)
    assert torch.equal(output[0], model(images)[0])
    zeros = torch.zeros(3, 8)
    assert ELBOLoss()((images, zeros, zeros), images) == 0
    assert torch.isfinite(ELBOLoss()(output, images))


def test_diffusion_forward_can_be_inverted_with_known_noise():
    diffusion = Diffusion(50)
    images = torch.randn(3, 1, 8, 8)
    steps = torch.tensor([0, 20, 49])
    corrupted, noise = diffusion.corrupt(images, steps)
    alpha = diffusion.alpha_bar[steps].view(-1, 1, 1, 1)
    recovered = (corrupted - (1 - alpha).sqrt() * noise) / alpha.sqrt()
    assert torch.allclose(recovered, images, atol=2e-6)
    generated = diffusion.sample(NoisePredictor(), 3)
    assert generated.shape == images.shape
    assert torch.isfinite(generated).all()


def test_gan_gradient_paths():
    generator, discriminator = Generator(), Discriminator()
    fake = generator(torch.randn(4, 16))
    discriminator(fake.detach()).sum().backward()
    assert generator.layers[0].weight.grad is None
    discriminator.zero_grad(set_to_none=True)
    for parameter in discriminator.parameters():
        parameter.requires_grad_(False)
    discriminator(generator(torch.randn(4, 16))).sum().backward()
    assert generator.layers[0].weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in discriminator.parameters())
