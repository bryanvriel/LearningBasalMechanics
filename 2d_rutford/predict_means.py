#!/usr/bin/env python3

# Globals
import numpy as np
import matplotlib.pyplot as plt
import iceutils as ice
from tqdm import tqdm
import sys
import os

# Tensorflow
import tensorflow as tf

# Locals
import networks
from utilities import *
from sample_training_data import make_design_matrix

AUXDIR = '/Volumes/USB/glaciers/rutford/aux'

def main():

    # Load full data grids
    with h5py.File('data_grids.h5', 'r') as fid:
        x = fid['x'][()]
        y = fid['y'][()]
        t = fid['tdec'][()]
        X, Y = np.meshgrid(x, y)
    Nt = len(t)
    Ny, Nx = X.shape

    # Load training data mask coordinates
    tx, ty = ice.load_kml(os.path.join(AUXDIR, 'rutford_withshelf_extrawide_training_mask.kml'),
                        out_epsg=3031)
    boundary = ice.Boundary(tx, ty)

    # Create mask
    train_mask = boundary.contains_points(X.ravel(), Y.ravel())

    # Create 1D points for prediction
    X_pts = X.ravel()[train_mask].reshape(-1, 1).astype(np.float32)
    Y_pts = Y.ravel()[train_mask].reshape(-1, 1).astype(np.float32)
    Npts = X_pts.size

    # Load bounds
    _, _, bounds = load_data()

    # Create model
    model = networks.IceStreamNet(bounds)

    # Create checkpoint objects for variables
    ckpt = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(ckpt, 'checkpoints', 1)
    ckpt.restore(manager.latest_checkpoint).expect_partial()
    
    # Convenience function to place 1d predictions into 2d array
    def make_2d_results(result1d):
        # Make empty array
        result2d = np.full((Ny, Nx), np.nan, dtype=np.float32)
        # Place 1d results in correct spot
        result2d.ravel()[train_mask] = result1d.numpy().squeeze()
        # Done
        return result2d

    @tf.function
    def predict(X, Y, G):

        # Predict un-normalized mean
        U, V, H, S, U_std, V_std, H_std, S_std = model.solution.call_inorm_full(X, Y, G)

        # Predict mean drag
        Tb = model.compute_drag(X, Y, G)
        #Tb = model.compute_drag_tfgrad(X, Y, G)

        return U, V, U_std, V_std, Tb, H, S, H_std, S_std

    # Create output file and datasets
    fid = h5py.File('output_means.h5', 'w')
    fid['x'] = x
    fid['y'] = y
    fid['tdec'] = t

    for key in ('U', 'V', 'H', 'S', 'Tb'):
        group = fid.create_group(key)
        if key in ('U', 'V'):
            group.create_dataset('mean', (Nt, Ny, Nx), 'f')
            group.create_dataset('std', (Nt, Ny, Nx), 'f')
        elif key in ('H', 'S'):
            group.create_dataset('mean', (Ny, Nx), 'f')
            group.create_dataset('std', (Ny, Nx), 'f')
        else:
            group.create_dataset('mean', (Nt, Ny, Nx), 'f')

    # Loop over time steps 
    for k in tqdm(range(Nt)):

        # Run prediction for current time step
        T_pts = np.full(X_pts.shape, t[k], dtype=np.float32)
        G = make_design_matrix(T_pts)
        values = predict(X_pts, Y_pts, G)
        
        # Unpack
        fid['U/mean'][k, :, :] = make_2d_results(values[0])
        fid['V/mean'][k, :, :] = make_2d_results(values[1])
        fid['U/std'][k, :, :] = make_2d_results(values[2])
        fid['V/std'][k, :, :] = make_2d_results(values[3])
        fid['Tb/mean'][k, :, :] = make_2d_results(values[4])
        if k == 0:
            fid['H/mean'][:, :] = make_2d_results(values[5])
            fid['S/mean'][:, :] = make_2d_results(values[6])
            fid['H/std'][:, :] = make_2d_results(values[7])
            fid['S/std'][:, :] = make_2d_results(values[8])


if __name__ == '__main__':
    main()

# end of file
