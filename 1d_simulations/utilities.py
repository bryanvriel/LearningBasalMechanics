#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
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

        # If using temporal model (e.g., periodic), use design matrix in place of time vector
        if loadG:
            T = fid['G'][()]
        else:
            T = fid['T'][()]

        # Normalize outputs
        U = Normalizer(*bounds['U'])(U)
        H = Normalizer(*bounds['H'])(H)

        #sc = plt.scatter(fid['T'][()].squeeze(), X.squeeze(), s=20, c=U.squeeze())
        #plt.colorbar(sc)
        #plt.show(); sys.exit()

        # Make solution data object
        data_fit = Data(train_fraction=0.85,
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
        data_pde = Data(train_fraction=0.85,
                        batch_size=batch_size,
                        shuffle=True,
                        split_seed=seed,
                        X=Xp, T=Tp)

    return data_fit, data_pde, bounds


def atleast_2d(x):
    """
    Convenience function to ensure arrays are column vectors.
    """
    if x.ndim == 1:
        return x.reshape(-1, 1)
    elif x.ndim == 2:
        return x
    else:
        raise NotImplementedError('Input array has greater than 2 dimensions')


def train_test_indices(N, train_fraction=0.9, shuffle=True, rng=None):
    """
    Convenience function to get train/test splits.
    """
    n_train = int(np.floor(train_fraction * N))
    if shuffle:
        assert rng is not None, 'Must pass in a random number generator'
        ind = rng.permutation(N)
    else:
        ind = np.arange(N, dtype=int)
    ind_train = ind[:n_train]
    ind_test = ind[n_train:]

    return ind_train, ind_test


class Data:
    """
    Class for representing and returning scattered points of solutions and coordinates.
    """

    def __init__(self, *args, train_fraction=0.9, train_indices=None, test_indices=None,
                 batch_size=1024, shuffle=True, seed=None, split_seed=None,
                 full_traversal=True, **kwargs):
        """
        Initialize dictionary of data and batching options. Data should be passed in
        via the kwargs dictionary.
        """
        # Check nothing has been passed in *args
        if len(args) > 0:
            raise ValueError('Data does not accept non-keyword arguments.')

        # Create a random number generator
        self.rng = np.random.RandomState(seed=seed)
        if split_seed is not None:
            self.split_rng = np.random.RandomState(seed=split_seed)
        else:
            self.split_rng = np.random.RandomState(seed=seed)

        # Cache first variable in order to get data shapes
        _first_key = next(iter(kwargs))
        self.n_data = kwargs[_first_key].shape[0]
            
        # Generate train/test indices if not provided explicitly
        self.shuffle = shuffle
        if train_indices is None or test_indices is None:
            itrain, itest = train_test_indices(self.n_data,
                                               train_fraction=train_fraction,
                                               shuffle=shuffle,
                                               rng=self.split_rng)
        else:
            itrain = train_indices
            itest = test_indices

        # Unpack the data for training
        self.keys = sorted(kwargs.keys())
        self._train = {}
        for key in self.keys:
            self._train[key] = kwargs[key][itrain]

        # Unpack the data for testing
        self._test = {}
        for key in self.keys:
            self._test[key] = kwargs[key][itest]

        # Cache training and batch size
        self.n_train = self._train[_first_key].shape[0]
        self.n_test = self.n_data - self.n_train
        self.batch_size = batch_size
        self.n_batches = int(np.ceil(self.n_train / self.batch_size))
        self.full_traversal = full_traversal

        # Initialize training indices (data have already been shuffle, so only need arange here)
        self._itrain = np.arange(self.n_train, dtype=int)

        # Initialize counter for training data retrieval
        self._train_counter = 0

        return

    def train_batch(self):
        """
        Get a random batch of training data as a dictionary. Ensure that we cycle through
        complete set of training data (e.g., sample without replacement)
        """
        # If self.full_traversal, we iterate over training indices without replacement
        if self.full_traversal:

            # If we've already reached the end of the training data, re-set counter with
            # optional re-shuffling of training indices
            if self._train_counter >= self.n_train:
                self.reset_training()

            # Construct slice for training data indices
            islice = slice(self._train_counter, self._train_counter + self.batch_size)
            indices = self._itrain[islice]

        # Otherwise, randomly choose from full set of training indices
        else:
            indices = self.rng.choice(self.n_train, size=self.batch_size, replace=False)

        # Get training data as a MultiVariable
        result = MultiVariable({key: self._train[key][indices] for key in self.keys})

        # Update counter for training data
        self._train_counter += self.batch_size

        return result

    def test_batch(self, batch_size=None):
        """
        Get a random batch of testing data as a dictionary.
        """
        batch_size = batch_size or self.batch_size
        ind = self.rng.choice(self.n_test, size=batch_size)
        return MultiVariable({key: self._test[key][ind] for key in self.keys})

    @property
    def train(self):
        """
        Get entire training set.
        """
        return self._train

    @train.setter
    def train(self, value):
        raise ValueError('Cannot set train variable.')

    @property
    def test(self):
        """
        Get entire testing set.
        """
        return self._test

    @test.setter
    def test(self, value):
        raise ValueError('Cannot set test variable.')

    def reset_training(self):
        """
        Public interface to reset training iteration counter and optionall
        re-shuffle traning indices
        """
        self._train_counter = 0
        if self.shuffle:
            self._itrain = self.rng.permutation(self.n_train)


class Normalizer:
    """
    Simple convenience class that performs transformations to/from normalized values.
    Here, we use the norm range [-1, 1] for pos=False or [0, 1] for pos=True.
    """

    def __init__(self, xmin, xmax, pos=False, log=False):
        self.xmin = xmin
        self.xmax = xmax
        self.denom = xmax - xmin
        self.pos = pos
        self.log = log
        self.log_eps = 0.05

    def __call__(self, x):
        """
        Alias for Normalizer.forward()
        """
        return self.forward(x)

    def forward(self, x):
        """
        Normalize data.
        """
        if self.pos:
            return (x - self.xmin) / self.denom
        elif self.log:
            xn = (x - self.xmin + self.log_eps) / self.denom
            return np.log(xn)
        else:
            return 2.0 * (x - self.xmin) / self.denom - 1.0

    def inverse(self, xn):
        """
        Un-normalize data.
        """
        if self.pos:
            return self.denom * xn + self.xmin
        elif self.log:
            return self.denom * np.exp(xn) + self.xmin - self.log_eps
        else:
            return 0.5 * self.denom * (xn + 1.0) + self.xmin

    def inverse_scale(self, scale, *args):
        """
        Un-normalize a scale factor (e.g., a standard deviation).
        """
        if self.pos:
            return self.denom * scale
        elif self.log:
            return (args[0] - self.xmin + self.log_eps) * scale
        else:
            return 0.5 * self.denom * scale


class MultiVariable(dict):
    """
    Class for representing multi-component input and output variables. Simple extension
    of Python dict with dot operator for accessing attributes and sum and concatenation
    methods. Note that since Python 3.7, dicts are ordered by insertion.
    """

    def __init__(self, *args, **kwargs):
        """
        Create N-d variables by dictionary unpacking or extracting columns from
        a column-stacked tensor.
        """
        # Special case: extract columns from tensor provided in positional argument
        if len(args) == 1 and isinstance(args[0], (tf.Tensor, np.ndarray)):

            # Get shape of tensor
            dat = args[0]
            N_batch, N_col = tf.shape(dat)

            # Loop over variables to get total number of dimensions
            ndims = 0
            for varname, ndim in kwargs.items():
                ndims += ndim
            assert ndims == N_col

            # Set variables
            for cnt, (varname, ndim) in enumerate(kwargs.items()):
                super().__setitem__(varname, tf.reshape(dat[:, cnt], (-1, ndim)))

        # Otherwise, init the parent dict
        else:
            super().__init__(*args, **kwargs)
        
    def concat(self, var_list=None):
        """
        Concatenates individual variables along the last dimension.
        """
        # List all variables
        if var_list is None:
            values = self.values()

        # Or specific variables
        else:
            values = [self.vars[name] for name in var_list]

        # Concatenate and return
        return tf.concat(values=values, axis=-1)

    def names(self):
        """
        Return the variable names.
        """
        return list(self.keys())

    def sum(self):
        """
        Returns sum over all variables. Tensorflow will broacast dimensions when possible.
        """
        return sum([value for value in self.vars.values()])

    def __getattr__(self, attr):
        """
        Return specific variable value using dot(.) operator.
        """
        return self.get(attr)

    def __setattr__(self, key, value):
        """
        Set specific variable value using dot(.) operator.
        """
        self.__setitem__(key, value)


# end of file
