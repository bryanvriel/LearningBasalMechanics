# LearningBasalMechanics
Python classes and scripts for the manuscript "Data-Driven Inference of the Mechanics of Slip Along Glacier Beds Using Physics-Informed Neural Networks: Case study on Rutford Ice Stream, Antarctica" by B. Riel, B. Minchew, and T. Bischoff (https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021MS002621). Included in this repository are scripts for training physics-informed neural networks for 1D and 2D ice flow. For both cases, ice flow follows the shallow shelf/stream approximation (SSA).

All neural network architectures used in the manuscript are implemented in `networks.py`, and training is implemented in `train*.py`. Utilities for loading and pre-processing training and validation data (from HDF5) can be found in `utilities.py`.

## Prerequisites
```
numpy
matplotlib
tensorflow
tensorflow_probability
pgan (https://github.com/bryanvriel/pgan)
tqdm
h5py
```
