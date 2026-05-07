# NIPS2026_ZOSGDMPR
Code for NIPS2026
## Requirements

The experiments are implemented in Python with PyTorch. We recommend using a CUDA-enabled GPU to reproduce the main experiments.

### Software dependencies

The main dependencies are:

```text
python >= 3.10
torch >= 2.0
torchvision >= 0.15
numpy
matplotlib
tqdm

We recommend creating a new conda environment:

conda create -n zosgdm-pr python=3.10 -y
conda activate zosgdm-pr

Then install the required packages:

pip install torch torchvision numpy matplotlib tqdm

Alternatively, you may install PyTorch following the official instructions for your CUDA version, and then install the remaining packages:

pip install numpy matplotlib tqdm
Dataset

The CIFAR-10 dataset is automatically downloaded by torchvision.datasets.CIFAR10 when running the experiment scripts. No manual dataset preparation is required.

Reproducing figures

The plotting scripts use saved result files and require only:

numpy
matplotlib

The generated figures will be saved to the output directory specified in the corresponding script.
