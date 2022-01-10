#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import h5py
import sys
import os

import iceutils as ice

def main():

    # Load samples
    x, t, U, H, U_samples, H_samples = ice.h5read(
        'output_predictions.h5',
        ['x', 't', 'U_mean', 'H_mean', 'U_smoothed_samples', 'H_smoothed_samples']
    )
    N_samples, Nx, Nt = U_samples.shape
    dx = abs(x[1] - x[0])

    # Compute drag mean using smoothed values of U and H
    Tb_mean = compute_drag(U, H, dx, prefilter=False)

    # Loop over samples and compute drag samples
    Tb_samples = np.zeros_like(U_samples)
    for k in tqdm(range(N_samples)):
        Tb_samples[k] = compute_drag(U_samples[k], H_samples[k], dx, prefilter=False)

    # Compute standard deviation
    Tb_std = np.std(Tb_samples, axis=0)
    arrs = (Tb_mean, Tb_std, Tb_samples)
    keys = ('tau_b_mean', 'tau_b_std', 'tau_b_samples')

    # Update predictions
    with h5py.File('output_predictions.h5', 'r+') as fid:
        for key, arr in zip(keys, arrs):
            try:
                del fid[key]
            except KeyError:
                pass
            fid[key] = arr


def compute_drag(U, H, dx, prefilter=False):

    n = 3
    AGlen = 3.827482556588247e-08 # AGlen(T = -5, KPa=True)
    eta_factor = 0.5 * AGlen**(-1.0 / n)
    B_x = -0.001
    rho_ice = 917.0
    rho_water = 1024.0
    g = 9.80665

    # Optional Gaussian filter (larger sigma along X-direction)
    if prefilter:
        U = gaussian_filter(U, sigma=(8.0, 0.1))
        H = gaussian_filter(H, sigma=(8.0, 0.1))
        #plt.imshow(U); plt.show(); sys.exit()

    # Gradients
    u_x = np.gradient(U, dx, axis=0, edge_order=2)

    # Stress
    strain = np.abs(u_x) + 1.0e-8
    eta = eta_factor * strain**((1.0 - n) / n)
    txx = 2.0 * eta * u_x

    # Membrane
    tm = np.gradient(2.0 * H * txx, dx, axis=0, edge_order=2)

    # Driving
    h_x = np.gradient(H, dx, axis=0, edge_order=2)
    s_x = h_x + B_x
    td = 1.0e-3 * rho_ice * g * H * s_x

    # Drag
    tb = -1.0 * (tm - td)

    return tb

if __name__ == '__main__':
    main()

# end of file
