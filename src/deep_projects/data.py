from pathlib import Path
import numpy as np
import pandas as pd
import torch


def ecg_data(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError("Download ECG first: python scripts/download_data.py ecg")
    frame = pd.read_csv(path, header=None)
    numeric = frame.apply(pd.to_numeric, errors="coerce").dropna()
    if numeric.shape[1] != 141:
        raise ValueError(f"Expected label and 140 amplitudes, received {numeric.shape[1]} columns")
    labels = torch.tensor(numeric.iloc[:, 0].to_numpy().astype(np.int64) - 1)
    signals = torch.tensor(numeric.iloc[:, 1:].to_numpy(), dtype=torch.float32)
    if labels.min() < 0 or labels.max() > 4:
        raise ValueError("ECG labels must be integers from 1 to 5")
    return signals, labels
