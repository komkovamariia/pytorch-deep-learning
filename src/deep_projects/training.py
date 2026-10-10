from dataclasses import dataclass
from pathlib import Path
import copy
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def seed_everything(seed: int = 1031) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class DataSplit:
    train: TensorDataset
    validation: TensorDataset
    test: TensorDataset

    def loaders(self, batch_size: int = 64) -> tuple:
        generator = torch.Generator().manual_seed(1031)
        return (
            DataLoader(self.train, batch_size, shuffle=True, generator=generator),
            DataLoader(self.validation, batch_size),
            DataLoader(self.test, batch_size),
        )


def split_tensors(features: torch.Tensor, labels: torch.Tensor) -> DataSplit:
    indices = np.arange(len(labels))
    development, test = train_test_split(indices, test_size=0.2, stratify=labels.numpy(), random_state=1031)
    train, validation = train_test_split(
        development, test_size=0.25, stratify=labels[development].numpy(), random_state=1031
    )
    return DataSplit(*(TensorDataset(features[i], labels[i]) for i in (train, validation, test)))


def digits_data(size: int = 16) -> DataSplit:
    digits = load_digits()
    images = torch.tensor(digits.images[:, None] / 16, dtype=torch.float32)
    images = nn.functional.interpolate(images, size=(size, size), mode="bilinear", align_corners=False)
    return split_tensors(images, torch.tensor(digits.target, dtype=torch.long))


@dataclass
class TrainConfig:
    epochs: int = 8
    learning_rate: float = 0.003
    weight_decay: float = 0.0
    device: str = "cpu"
    clip_norm: float = 5.0


class Trainer:
    def __init__(self, model: nn.Module, loss: nn.Module, config: TrainConfig):
        self.model = model.to(config.device)
        self.loss = loss
        self.config = config
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total, count = 0.0, 0
        for features, targets in loader:
            features, targets = features.to(self.config.device), targets.to(self.config.device)
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.loss(self.model(features), targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Training loss is not finite")
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_norm)
            self.optimizer.step()
            total += loss.item() * len(targets)
            count += len(targets)
        if count == 0:
            raise ValueError("Training loader is empty")
        return total / count

    @torch.inference_mode()
    def evaluate_loss(self, loader: DataLoader) -> float:
        self.model.eval()
        total, count = 0.0, 0
        for features, targets in loader:
            features, targets = features.to(self.config.device), targets.to(self.config.device)
            total += self.loss(self.model(features), targets).item() * len(targets)
            count += len(targets)
        if count == 0:
            raise ValueError("Evaluation loader is empty")
        return total / count

    def fit(self, train: DataLoader, validation: DataLoader) -> pd.DataFrame:
        if self.config.epochs < 1:
            raise ValueError("epochs must be positive")
        best_loss, best_weights = float("inf"), None
        history = []
        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(train)
            validation_loss = self.evaluate_loss(validation)
            history.append({"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": validation_loss})
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_weights = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(best_weights)
        self.model.eval()
        return pd.DataFrame(history)


@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    device = next(model.parameters()).device
    predictions, targets = [], []
    for features, labels in loader:
        predictions.append(model(features.to(device)).cpu())
        targets.append(labels)
    return torch.cat(predictions), torch.cat(targets)


def classification_scores(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    probabilities = logits.softmax(-1).numpy()
    predicted = probabilities.argmax(-1)
    return {
        "accuracy": float(accuracy_score(targets, predicted)),
        "macro_f1": float(f1_score(targets, predicted, average="macro", zero_division=0)),
        "nll": float(log_loss(targets, probabilities, labels=np.arange(logits.shape[-1]))),
    }


def plot_history(history: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(6, 3.5))
    axis.plot(history.epoch, history.train_loss, label="train")
    axis.plot(history.epoch, history.validation_loss, label="validation")
    axis.set(xlabel="Epoch", ylabel="Loss")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.show()


def image_grid(images: torch.Tensor, path: Path, titles=None) -> None:
    images = images.detach().cpu()
    count = min(len(images), 8)
    fig, axes = plt.subplots(1, count, figsize=(1.8 * count, 2.2), squeeze=False)
    for index, axis in enumerate(axes.flat):
        image = images[index]
        image = image.squeeze(0).numpy() if image.shape[0] == 1 else image.permute(1, 2, 0).numpy()
        axis.imshow(image, cmap="gray", vmin=0, vmax=1)
        axis.axis("off")
        if titles is not None:
            axis.set_title(str(titles[index]), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.show()
