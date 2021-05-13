#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
import pgan
import sys
import os

def compute_bounds(x, n_sigma=1.0, method='normal'):

    if method == 'normal':
        mean = np.mean(x)
        std = np.std(x)
        lower = mean - n_sigma * std
        upper = mean + n_sigma * std
        return [lower, upper]
    elif method == 'minmax':
        maxval = np.nanmax(x)
        minval = np.nanmin(x)
        return [minval, maxval]
    else:
        raise ValueError('Unsupported bounds determination method')
    
def get_bounds(filename):
    """
    Convenience method to compute lower and upper Normalization bounds to ensure
    zero mean and unit variance.
    """
    # Compute bounds for output and input variables
    bounds = {}
    with h5py.File(filename, 'r') as fid:
        for key in ('U', 'H'):
            value = fid[key][()]
            bounds[key] = compute_bounds(value, method='normal')
        for key in ('X', 'T'):
            value = fid[key][()]
            bounds[key] = compute_bounds(value, method='minmax')

    # Done
    return bounds

def load_data(seed=13, batch_size=128, loadG=False):
    """
    Each HDF5 group will correspond to a data instance.
    """
    # Set random state
    rng = np.random.RandomState(seed=seed)

    # Get data bounds
    bounds = get_bounds('data.h5')

    # Load gridded data
    with h5py.File('data.h5', 'r') as fid:

        # Load grid data
        U = fid['U'][()]
        H = fid['H'][()]
        X = fid['X'][()]

        # If using temporal model, use design matrix in place of time vector
        if loadG:
            T = fid['G'][()]
        else:
            T = fid['T'][()]

        # Normalize outputs
        U = pgan.data.Normalizer(*bounds['U'])(U)
        H = pgan.data.Normalizer(*bounds['H'])(H)

        # Make solution data object
        data_fit = pgan.data.Data(train_fraction=0.85,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  split_seed=seed,
                                  X=X, T=T, U=U, H=H)

        # Make basal drag data object
        Xp = fid['Xp'][()]
        if loadG:
            Tp = fid['Gp'][()]
        else:
            Tp = fid['Tp'][()]
        data_pde = pgan.data.Data(train_fraction=0.85,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  split_seed=seed,
                                  X=Xp, T=Tp)

    return data_fit, data_pde, bounds


# end of file
