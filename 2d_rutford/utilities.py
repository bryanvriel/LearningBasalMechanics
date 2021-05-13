#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import h5py
import pgan
import sys
import os

import tensorflow as tf

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

def get_bounds(filename, keys=['U', 'V', 'H', 'S', 'X', 'Y', 'T'], method='normal'):
    """
    Convenience method to compute lower and upper Normalization bounds to ensure
    zero mean and unit variance.
    """
    # Compute bounds for output and input variables
    bounds = {}
    with h5py.File(filename, 'r') as fid:
        for key in keys:
            value = fid[key][()]
            bounds[key] = compute_bounds(value, method=method)

    # Done
    return bounds

def load_data(filename='data.h5', seed=24, batch_size=512):
    """
    Each HDF5 group will correspond to a data instance.
    """
    # Set fixed random state
    rng = np.random.RandomState(seed=seed)

    # Get data bounds for output variables
    input_bounds = get_bounds(filename, keys=['X', 'Y'], method='minmax')
    bounds = get_bounds(filename, keys=['U', 'V', 'H', 'S'])
    bounds.update(input_bounds)

    # Load gridded data
    with h5py.File(filename, 'r') as fid:

        # Load grid data
        U = fid['U'][()]
        V = fid['V'][()]
        H = fid['H'][()]
        S = fid['S'][()]
        X = fid['X'][()]
        Y = fid['Y'][()]
        G = fid['G'][()]
        N_total = G.shape[0]

        # Normalize outputs
        U = pgan.data.Normalizer(*bounds['U'])(U)
        V = pgan.data.Normalizer(*bounds['V'])(V)
        H = pgan.data.Normalizer(*bounds['H'])(H)
        S = pgan.data.Normalizer(*bounds['S'])(S)

        # Use data for solution data object
        data_fit = pgan.data.Data(train_fraction=0.85,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  split_seed=seed,
                                  X=X, Y=Y, T=G, U=U, V=V, H=H, S=S)

        # Make basal drag data object
        Xpde, Ypde, Gpde = [fid[key][()] for key in ('Xpde', 'Ypde', 'Gpde')]
        data_pde = pgan.data.Data(train_fraction=0.85,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  split_seed=seed,
                                  X=Xpde, Y=Ypde, T=Gpde)

    return data_fit, data_pde, bounds


# end of file
