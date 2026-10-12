# Execution and validation

All 11 project notebooks were executed with `PROJECT_FULL=1` on CPU using seed 1031. The standard supervised budget was 40 epochs. The optimization comparison used 80, the ECG classifier used 30, the VAE used 80, the GAN used 150 and diffusion used 100 where those projects are present. Each notebook states its actual budget in the first output. Total recorded execution time for this repository was 202.4 seconds; hardware contention makes timing descriptive.

The runtime blocked local Jupyter kernel sockets. Execution therefore used the included plain-Python engine with Matplotlib's Agg backend. It ran the same cells in order and captured their stdout and figures into the notebook. Use `PROJECT_FULL=1 python scripts/execute_notebooks.py --inprocess` to repeat this route. Normal Jupyter execution is available by omitting `--inprocess`.

`python -m ruff check src tests scripts` passed, and all 6 unit tests passed after code formatting. The notebooks also check numerical agreement, tensor shapes and their topic-specific invariants. Every result JSON was produced by executed code with non-finite values rejected. The figures were visually inspected, including ECG reconstructions and generative sample grids.

Package versions are in `docs/environment.json` and `requirements-tested.txt`. `results/execution_report.json` records each executed notebook. Metrics measure the datasets and splits described in the notebooks. ECG scores use beat-level splits and do not establish generalization to unseen patients.
