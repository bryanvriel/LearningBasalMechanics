#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.signal import filtfilt, savgol_filter, convolve
from scipy.signal.windows import triang
import h5py
import sys
import os

import iceutils as ice

"""
The samples generates from dist.sample() are spatially noisy. Let's apply a 'light'
spline filter to each sample in order to get better stresses and stress gradients.
"""

def normalize(x):
    xmin, xmax = np.min(x), np.max(x)
    xn = (x - xmin) / (xmax - xmin)
    return xn, xmin, xmax

def unnormalize(xn, xmin, xmax):
    x = xn * (xmax - xmin) + xmin
    return x

def main():

    with h5py.File('output_predictions.h5', 'r') as fid:
        x = fid['x'][()]
        t = fid['t'][()]
        U_samples = fid['U_samples'][()]
        H_samples = fid['H_samples'][()]
    N_samples, Nx, Nt = U_samples.shape
    xn = 1.0e-5 * x

    dx = abs(x[1] - x[0])
    svals = [0.01, 0.125]
    sigma = 6.0
    butter_b, butter_a = ice.tseries.butterworth_coeffs(period=3.0e3, dt=dx)

    x_ext = np.hstack((
        x[0] + dx * np.arange(-int(2.5*sigma), 0, dtype=int),
        x,
        x[-1] + dx * np.arange(1, int(2.5*sigma), dtype=int)
    ))
    xn_ext = 1.0e-5 * x_ext

    x_norm = (x - x[0]) / (x[1] - x[0])
    t_year = 2000.0 + x_norm
    tmodel = ice.tseries.build_temporal_model(t_year, poly=1, periods=[],
                                              isplines=[64,])
    G = tmodel.G
    regDiag = np.zeros(G.shape[1])
    regDiag[2:] = 100.0
    GtG = np.dot(G.T, G) + np.diag(regDiag)

    method = 'bspline'

    def smoothe_profile(z, method='spline', s=20.0):

        # Normalize the data for numerical stability
        zn, zmin, zmax = normalize(z)

        # Interpolate with spline
        if method == 'spline':
            spl = UnivariateSpline(xn, zn, k=5, s=s)
            z_filt = unnormalize(spl(xn), zmin, zmax)

        elif method == 'bspline':
            Gtd = np.dot(G.T, zn)
            m = np.linalg.lstsq(GtG, Gtd, rcond=1.0e-12)[0]
            z_filt = unnormalize(np.dot(G, m), zmin, zmax)

        elif method == 'gaussian':
            zn_ext = interp1d(xn, zn, kind='linear', fill_value='extrapolate')(xn_ext)
            zn_filt = gaussian_filter1d(zn_ext, sigma=sigma, mode='nearest')
            zn_filt = interp1d(xn_ext, zn_filt, kind='linear')(xn)
            z_filt = unnormalize(zn_filt, zmin, zmax)

        elif method == 'convolve':
            zn_ext = interp1d(xn, zn, kind='linear', fill_value='extrapolate')(xn_ext)
            tfilt = triang(41)
            tfilt /= np.sum(tfilt)
            zn_filt = convolve(zn_ext, tfilt, mode='same')
            zn_filt = interp1d(xn_ext, zn_filt, kind='linear')(xn)
            z_filt = unnormalize(zn_filt, zmin, zmax)

        elif method == 'butter':
            zn_ext = interp1d(xn, zn, kind='linear', fill_value='extrapolate')(xn_ext)
            zn_filt = filtfilt(butter_b, butter_a, zn_ext, padtype='constant')
            zn_filt = interp1d(xn_ext, zn_filt, kind='linear')(xn)
            z_filt = unnormalize(zn_filt, zmin, zmax)

        else:
            raise ValueError

        return z_filt

    debug = False
    if debug:

        for cnt, samples in enumerate((U_samples, H_samples)):

            print(cnt)

            # Get means
            means = np.mean(samples, axis=0)

            # Loop over coefficients to test smoothing
            for j in range(0, Nt, 200):
                z = means[:, j]
                zf = smoothe_profile(z, method=method, s=svals[cnt])
                fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10, 5))
                ax1.plot(x, z)
                ax1.plot(x, zf)
                ax2.plot(x, np.gradient(zf))
                ax3 = ax2.twinx()
                ax3.plot(x, z - zf, color='C1')
                plt.tight_layout()
                plt.show()

        sys.exit()

    def smoothe_samples(samples, s=1.0):

        print('smoothing')

        # Allocate array for output
        smoothed = ice.pymp.zeros(samples.shape, dtype=np.float32)

        # Process samples
        with ice.pymp.Parallel(5) as parallel:
            for k in tqdm(parallel.range(N_samples)):

                # Get 2D array for current sample
                Z = samples[k, :, :]

                # Loop over time indices
                for j in range(Nt):
                    smoothed[k, :, j] = smoothe_profile(Z[:, j], method=method, s=s)
                
        return smoothed

    # Run smoothing over each variable
    U_smooth = smoothe_samples(U_samples, s=svals[0])
    H_smooth = smoothe_samples(H_samples, s=svals[1])

    # Store final results
    with h5py.File('output_predictions.h5', 'r+') as fid:
        for key in ('U_smoothed_samples', 'H_smoothed_samples'):
            try:
                del fid[key]
            except KeyError:
                pass
        fid['U_smoothed_samples'] = U_smooth
        fid['H_smoothed_samples'] = H_smooth


if __name__ == '__main__':
    main()

# end of file
