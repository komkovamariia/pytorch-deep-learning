# pytorch deep learning

These experiments follow neural-network training from explicit derivatives to recurrent, graph, spiking and generative models. Image and signal tasks provide concrete inputs for comparing models. The projects include real ECG5000 data from ML4HD and smaller controlled experiments that make the numerical behavior easier to inspect.

## Projects in reading order

| Order | Project |
| --- | --- |
| 01 | [Backpropagation from derivatives to autograd](notebooks/01_autograd_and_backprop.ipynb) |
| 02 | [Optimization and regularization on noisy image features](notebooks/02_optimization_and_regularization.ipynb) |
| 03 | [ECG compression with PCA and a learned bottleneck](notebooks/03_ecg_compression.ipynb) |
| 04 | [Reconstruction-based ECG anomaly detection](notebooks/04_ecg_anomaly_detection.ipynb) |
| 05 | [Forecasting a noisy sensor series](notebooks/05_recurrent_forecasting.ipynb) |
| 06 | [Classifying ECG beats with convolution and recurrence](notebooks/06_ecg_sequence_classification.ipynb) |
| 07 | [Image classification through graph message passing](notebooks/07_graph_message_passing.ipynb) |
| 08 | [A leaky integrate-and-fire classifier](notebooks/08_spiking_network.ipynb) |
| 09 | [A variational autoencoder for digit images](notebooks/09_variational_autoencoder.ipynb) |
| 10 | [Training a small image GAN](notebooks/10_adversarial_generation.ipynb) |
| 11 | [Noise prediction and reverse diffusion](notebooks/11_denoising_diffusion.ipynb) |

The early notebooks establish useful baselines. Later projects add new model assumptions and compare them against references where a common task allows it. Read each notebook's data and evaluation notes before comparing its scores with another project.

## Run locally

Use Python 3.10 or newer and a matched PyTorch and torchvision installation. For CPU-only installation, install those two packages from PyTorch's CPU wheel index first.

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
python scripts/download_data.py ecg
python -m pytest -q
python scripts/execute_notebooks.py
```

Open the notebooks with JupyterLab from this repository. They find the repository root and use the package in `src`. The default budget is short and runs on CPU. The committed results were produced with `PROJECT_FULL=1`. Set `PROJECT_FULL=1` in the environment for longer training.

## Recorded runs

The notebook outputs, [figures](figures) and [result files](results) come from executed full runs. [Execution details](docs/validation.md) record the environment, checks and limits. Generator sample grids are visual inspections; they are not presented as benchmark-quality generation results. New experiments must be executed again to produce their measurements.

Training and validation use separate subsets. Validation selects checkpoints and thresholds. Final test metrics are recorded after those choices. In the ECG projects, available data support a beat-level split; unseen-patient generalization is not measured.

## Code and sources

Models and reusable training code live in `src`. Tests cover tensor shapes, numerical invariants and saved-model agreement. `python -m ruff check src tests scripts` checks unused names and common errors.

The project sequence draws on UniPd Machine Learning for Human Data, UniPd Computer Vision and Cognitive Systems and HSE deep learning exercises. [Source mapping](docs/sources.md) connects each project to the specific course topic. These are independent implementations. Original course solutions, datasets and saved teaching-model weights are not redistributed here. Data access is documented in [datasets.md](docs/datasets.md).
