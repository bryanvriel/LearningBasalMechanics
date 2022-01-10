## Summary

This example demonstrates the use of PINNs for inferring basal drag for a 1D laterally-confined ice stream. Here, we use synthetic time-dependent velocity and ice thickness data generated from ice flow simulations, which use the shallow stream approximation (SSA) for the momentum balance. In this example, we train a neural network to reconstruct the time-dependent velocity and thickness. The loss function will consist of a data reconstruction loss and a physics-based loss that places constraints on the inferred basal drag (see [Riel et al., 2021, JAMES](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021MS002621)).

The noise-free, gridded simulation data is in the HDF5 file `data_periodic.h5`. For a spatial domain of size `Nx = 2445` and number of time steps `Nt = 500`, the relevant contents of the file are:

```
B                        Dataset {500}        # bed (meters)
B_X                      Dataset {500}        # bed slope
H                        Dataset {2445, 500}  # ice thickness (meters)
T                        Dataset {2445}       # time (years)
U                        Dataset {2445, 500}  # velocity (m/yr)
X                        Dataset {500}        # X-coordinates of profile (meters)
```

For this example, we have added a combination of white noise and spatially-correlated noise (see manuscript for technical details). The gridded data with the noise added is contained in the file `data_grids.h5`. For training purposes, we have randomly sampled 100000 points from the gridded, noisy velocity and thickness data. These data are stored in the file `data.h5`:

```
B_x                      Dataset {100000, 1}  # bed slope
G                        Dataset {100000, 6}  # temporal design matrix
Gp                       Dataset {100000, 6}  # temporal design matrix for physics loss
H                        Dataset {100000, 1}  # thickness
T                        Dataset {100000, 1}  # time
Tp                       Dataset {100000, 1}  # time for physics loss
U                        Dataset {100000, 1}  # velocity
X                        Dataset {100000, 1}  # X-coordinate
Xp                       Dataset {100000, 1}  # X-coordinate for physics loss
```
Note that in addition to the randomly sampled data, we also include an independent set of random coordinates for computing the physics-based loss. Additionally, we construct a temporal design matrix `G` at the randomly sampled time points which we use to predict time-dependent variables, e.g. `u(x, t) = G(t) * m(x)` where `m` is a vector of coefficients at a given spatial coordinate (see manuscript for details).

### Step 1: Pre-training

For the first step, we will train a feedforward neural network to reconstruct the velocity and thickness observations using only a standard MSE-type loss (in actuality, we use a log-likelihood loss by parameterizing the velocity and thickness as independent Normal distributions). This step is not necessary, but we found it to improve training for the combined loss. Pre-training is run as:

```
./pretrain.py
```

### Step 2: Training

For the next step, we will train the same neural network using the combined data reconstruction and physics-based loss:

```
./train.py
```

### Step 3: Predict `U` and `H` samples

After training, we will generate predictions of velocity and ice thickness over the entire simulation grid. Instead of predicting single values of `U` and `H`, we will generate random samples since our network predicts mean and standard deviation.

```
./predict.py
```
This script will create a file called `output_predictions.h5`:

```
H_mean                   Dataset {500, 1223}
H_samples                Dataset {100, 500, 1223}
H_std                    Dataset {500, 1223}
U_mean                   Dataset {500, 1223}
U_samples                Dataset {100, 500, 1223}
U_std                    Dataset {500, 1223}
t                        Dataset {1223}
x                        Dataset {500}
```

From the dataset sizes, we can see that we have generated 100 random samples of the full space-time volume. Note that since we parameterize normal distributions that are spatially independent, these samples will have unrealistic high-frequency noise, which would result in un-physical ice stresses. A better approach would be to predict `U` and `H` over a finite area using a multivariate distribution with some notion of spatial correlation (work in progress). For now, we will perform a post-processing step where we spatially smoothe the samples using a small smoothing window:

```
./smoothe_samples.py
```
This script will create two additional datasets in `output_predictions.h5`:

```
H_smoothed_samples       Dataset {100, 500, 1223}
U_smoothed_samples       Dataset {100, 500, 1223}
```

### Step 4: Generate drag samples

Using the smoothed samples from the previous step, we can now generate samples of time-dependent basal drag:

```
./generate_drag_samples.py
```
Within this script is a function called `compute_drag()` that shows the form of the SSA momentum balance used to compute drag. After this script is run, we will have three additional datasets in `output_predictions.h5` that store our drag predictions:

```
tau_b_mean               Dataset {500, 1223}
tau_b_samples            Dataset {100, 500, 1223}
tau_b_std                Dataset {500, 1223}
```
