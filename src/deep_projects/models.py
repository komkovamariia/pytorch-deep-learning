import math
import torch
from torch import nn
from torch.nn import functional as F


class DenseAutoencoder(nn.Module):
    def __init__(self, features=140, latent=12):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(features, 64), nn.ReLU(), nn.Linear(64, latent))
        self.decoder = nn.Sequential(nn.Linear(latent, 64), nn.ReLU(), nn.Linear(64, features))

    def forward(self, signals):
        return self.decoder(self.encoder(signals))


class RecurrentForecaster(nn.Module):
    def __init__(self, cell="gru", hidden=24):
        super().__init__()
        recurrent = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[cell]
        self.recurrent = recurrent(1, hidden, batch_first=True)
        self.readout = nn.Linear(hidden, 1)

    def forward(self, windows):
        sequence, _ = self.recurrent(windows)
        return self.readout(sequence[:, -1])


class SignalGRU(nn.Module):
    def __init__(self, classes=5):
        super().__init__()
        self.recurrent = nn.GRU(1, 24, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(48, classes)

    def forward(self, signals):
        _, hidden = self.recurrent(signals.unsqueeze(-1))
        return self.classifier(torch.cat([hidden[-2], hidden[-1]], dim=-1))


class SignalCNN(nn.Module):
    def __init__(self, classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(256, classes)

    def forward(self, signals):
        return self.classifier(self.features(signals.unsqueeze(1)))


def grid_adjacency(side: int = 8) -> torch.Tensor:
    adjacency = torch.eye(side * side)
    for row in range(side):
        for column in range(side):
            node = row * side + column
            for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                nr, nc = row + dr, column + dc
                if 0 <= nr < side and 0 <= nc < side:
                    adjacency[node, nr * side + nc] = 1
    inverse_degree = adjacency.sum(-1).rsqrt()
    return inverse_degree[:, None] * adjacency * inverse_degree[None, :]


class GridGCN(nn.Module):
    def __init__(self, adjacency: torch.Tensor):
        super().__init__()
        self.register_buffer("adjacency", adjacency)
        side = math.isqrt(len(adjacency))
        rows, columns = torch.meshgrid(
            torch.linspace(-1, 1, side), torch.linspace(-1, 1, side), indexing="ij"
        )
        self.register_buffer("positions", torch.stack([rows.flatten(), columns.flatten()], dim=-1))
        self.first = nn.Linear(3, 24)
        self.second = nn.Linear(24, 24)
        self.classifier = nn.Linear(len(adjacency) * 24, 10)

    def forward(self, images):
        intensities = images.flatten(2).transpose(1, 2)
        positions = self.positions.expand(len(images), -1, -1)
        features = torch.cat([intensities, positions], dim=-1)
        hidden = F.relu(self.first(self.adjacency @ features))
        hidden = F.relu(self.second(self.adjacency @ hidden))
        return self.classifier(hidden.flatten(1))


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, voltage):
        ctx.save_for_backward(voltage)
        return (voltage > 0).to(voltage.dtype)

    @staticmethod
    def backward(ctx, gradient):
        (voltage,) = ctx.saved_tensors
        return gradient / (1 + 5 * voltage.abs()).square()


class SpikingMLP(nn.Module):
    def __init__(self, steps=12):
        super().__init__()
        self.input_layer = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, 10)
        self.steps = steps
        self.decay = 0.9

    def forward(self, images):
        intensities = images.flatten(1)
        membrane = intensities.new_zeros(len(images), 64)
        output = intensities.new_zeros(len(images), 10)
        for _ in range(self.steps):
            # Deterministic current encoding keeps evaluation repeatable.
            membrane = self.decay * membrane + self.input_layer(intensities)
            spikes = SurrogateSpike.apply(membrane - 1)
            membrane = membrane - spikes.detach()
            output = output + self.output_layer(spikes)
        return output / self.steps


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent=8):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        self.mean = nn.Linear(64, latent)
        self.log_variance = nn.Linear(64, latent)
        self.decoder = nn.Sequential(nn.Linear(latent, 64), nn.ReLU(), nn.Linear(64, 64), nn.Sigmoid())

    def encode(self, images):
        hidden = self.encoder(images.flatten(1))
        return self.mean(hidden), self.log_variance(hidden)

    def forward(self, images):
        mean, log_variance = self.encode(images)
        latent = mean + torch.exp(0.5 * log_variance) * torch.randn_like(mean) if self.training else mean
        return self.decoder(latent).view_as(images), mean, log_variance


class ELBOLoss(nn.Module):
    def __init__(self, beta=0.05):
        super().__init__()
        self.beta = beta

    def forward(self, output, target):
        reconstructed, mean, log_variance = output
        reconstruction = (reconstructed - target).square().flatten(1).sum(1).mean()
        divergence = -0.5 * (1 + log_variance - mean.square() - log_variance.exp()).sum(1).mean()
        return reconstruction + self.beta * divergence


class Generator(nn.Module):
    def __init__(self, latent=16):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.Tanh(),
        )

    def forward(self, noise):
        return self.layers(noise).view(-1, 1, 8, 8)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Flatten(), nn.Linear(64, 64), nn.LeakyReLU(0.2), nn.Linear(64, 1))

    def forward(self, images):
        return self.layers(images)


class NoisePredictor(nn.Module):
    def __init__(self, steps=50):
        super().__init__()
        self.steps = steps
        self.layers = nn.Sequential(
            nn.Linear(65, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 64)
        )

    def forward(self, noisy_images, timesteps):
        inputs = torch.cat([noisy_images.flatten(1), timesteps[:, None] / self.steps], dim=-1)
        return self.layers(inputs).view_as(noisy_images)


class Diffusion:
    def __init__(self, steps=50):
        self.beta = torch.linspace(0.0001, 0.12, steps)
        self.alpha = 1 - self.beta
        self.alpha_bar = self.alpha.cumprod(0)

    def corrupt(self, images, timesteps):
        noise = torch.randn_like(images)
        cumulative_alpha = self.alpha_bar[timesteps].view(-1, 1, 1, 1)
        return cumulative_alpha.sqrt() * images + (1 - cumulative_alpha).sqrt() * noise, noise

    @torch.inference_mode()
    def sample(self, model, count=8):
        model.eval()
        images = torch.randn(count, 1, 8, 8)
        for step in range(len(self.beta) - 1, -1, -1):
            timesteps = torch.full((count,), step, dtype=torch.long)
            predicted_noise = model(images, timesteps)
            alpha_bar = self.alpha_bar[step]
            clean_estimate = (images - (1 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt()
            # The training images lie in [-1, 1]; clipping constrains the estimated clean image.
            clean_estimate = clean_estimate.clamp(-1, 1)
            previous_alpha_bar = self.alpha_bar[step - 1] if step > 0 else torch.tensor(1.0)
            clean_coefficient = self.beta[step] * previous_alpha_bar.sqrt() / (1 - alpha_bar)
            noisy_coefficient = self.alpha[step].sqrt() * (1 - previous_alpha_bar) / (1 - alpha_bar)
            mean = clean_coefficient * clean_estimate + noisy_coefficient * images
            if step > 0:
                variance = self.beta[step] * (1 - self.alpha_bar[step - 1]) / (1 - self.alpha_bar[step])
                images = mean + variance.sqrt() * torch.randn_like(images)
            else:
                images = mean
        return images
