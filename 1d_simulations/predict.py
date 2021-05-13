#!/usr/bin/env python3

# Globals
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import sys
import os

# Tensorflow
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

# Locals
import networks
from utilities import *
from pretrain import USE_TEMPORAL

def main():

    # Load full data grids
    with h5py.File('data_grids.h5', 'r') as fid:
        x = fid['X'][()]
        t = fid['T'][()][::2]
        X, T = np.meshgrid(x, t)
        Nt, Nx = X.shape

    # Load bounds
    _, _, bounds = load_data()

    # Create model
    model = networks.IceStreamNet(bounds, temporal_model=USE_TEMPORAL)

    # Create checkpoint objects for variables
    ckpt = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(ckpt, 'checkpoints', 1)
    ckpt.restore(manager.latest_checkpoint).expect_partial()

    @tf.function
    def predict_mean(X, T):
        U_mean, H_mean = model.solution.call_inorm(X, T)
        return U_mean, H_mean

    @tf.function
    def predict_samples(X, T):

        # Predict likelihoods
        u_dist, h_dist = model.solution(X, T)

        # Generate sample
        u = u_dist.sample()
        h = h_dist.sample()

        # Un-normalize
        u = model.solution.Un.inverse(u)
        h = model.solution.Hn.inverse(h)

        return u, h

    # Allocate arrays for predictions
    n_samples = 100
    U_mean = np.zeros((Nx, Nt), dtype=np.float32)
    H_mean = np.zeros((Nx, Nt), dtype=np.float32)
    U_samples = np.zeros((n_samples, Nx, Nt), dtype=np.float32)
    H_samples = np.zeros((n_samples, Nx, Nt), dtype=np.float32)

    # Generate samples by looping over X batches
    batch_size = 16
    n_batches = int(np.ceil(Nx / batch_size))
    for j in tqdm(range(n_batches)):

        # Create inputs
        xslice = slice(j * batch_size, (j + 1) * batch_size)
        xb = x[xslice]
        Nx = xb.size
        Tb, Xb = [arr.reshape(-1, 1).astype(np.float32) for arr in np.meshgrid(t, xb)]

        if USE_TEMPORAL:
            ω1 = 2.0 * np.pi / 0.5
            ω2 = 2.0 * np.pi / 1.0
            Tb = np.column_stack((
                np.ones_like(Tb),
                Tb,
                np.cos(ω1 * Tb),
                np.sin(ω1 * Tb),
                np.cos(ω2 * Tb),
                np.sin(ω2 * Tb),
            ))

        # Predict means
        um, hm = predict_mean(Xb, Tb)
        U_mean[xslice, :] = um.numpy().reshape(Nx, Nt)
        H_mean[xslice, :] = hm.numpy().reshape(Nx, Nt)

        # Loop over n_samples
        for k in range(n_samples):
            us, hs = predict_samples(Xb, Tb)
            U_samples[k, xslice, :] = us.numpy().reshape(Nx, Nt)
            H_samples[k, xslice, :] = hs.numpy().reshape(Nx, Nt)

    # Compute standard deviations for everything
    U_std = np.std(U_samples, axis=0)
    H_std = np.std(H_samples, axis=0)
    arrs = (U_mean, U_std, H_mean, H_std)
    keys = ('U_mean', 'U_std', 'H_mean', 'H_std')
        
    # Store final results
    with h5py.File('output_predictions.h5', 'w') as fid:
        fid['x'] = x
        fid['t'] = t
        fid['U_samples'] = U_samples
        fid['H_samples'] = H_samples
        for key, arr in zip(keys, arrs):
            fid[key] = arr
            

if __name__ == '__main__':
    main()

# end of file
