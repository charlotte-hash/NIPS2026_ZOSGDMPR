# NIPS2026_ZOSGDMPR

Code for NIPS2026.

## Requirements

The experiments are implemented in Python with PyTorch. We recommend using a CUDA-enabled GPU to reproduce the main experiments.

## Software Dependencies

The main dependencies are:

- Python >= 3.10
- PyTorch >= 2.0
- torchvision >= 0.15
- NumPy
- Matplotlib
- tqdm

A CUDA-compatible PyTorch installation is recommended for GPU acceleration. Please install the PyTorch version that matches your local CUDA environment.

## Installation

We recommend creating a new conda environment.

```bash
conda create -n zosgdm-pr python=3.10 -y
conda activate zosgdm-pr
```

Then install the required packages.

```bash
pip install torch torchvision numpy matplotlib tqdm
```

Alternatively, you may install PyTorch following the official instructions for your CUDA version, and then install the remaining packages.

```bash
pip install numpy matplotlib tqdm
```

## Dataset

The CIFAR-10 dataset is automatically downloaded by `torchvision.datasets.CIFAR10` when running the experiment scripts. No manual dataset preparation is required.

## Reproducing Figures

The plotting scripts use saved result files and require only NumPy and Matplotlib.

The generated figures will be saved to the output directory specified in the corresponding script.
