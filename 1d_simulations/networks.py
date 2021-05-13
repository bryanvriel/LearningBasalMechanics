#-*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import pgan
import sys


def unpack_parameters(par):
    """
    Unpacks a tensor of shape (N, 2) into mean and standard deviation.
    """
    mean = tf.expand_dims(par[:, 0], axis=1)
    std = 1.0e-4 + 0.01 * tf.nn.softplus(tf.expand_dims(par[:, 1], axis=1))
    dist = tfd.Normal(loc=mean, scale=std)
    return dist


def unpack_coeff_parameters(par, G, Npar):
    """
    Unpacks a tensor of shape (N, 6) into mean and standard deviation for each 
    coefficient of the function:

        u(t) = a * cos(ωt) + b * sin(ωt) + c

    Return only the ::predicted:: mean and standard deviation using
    propagation of uncertainties.
    """
    # Partition the mean and standard deviations
    phi_mean = par[:, :Npar]
    phi_std = 1.0e-4 + 0.01 * tf.nn.softplus(par[:, Npar:])

    # Multiply by design matrix
    mean = tf.reduce_sum(G * phi_mean, axis=1, keepdims=True)

    # Compute standard deviation
    std = tf.sqrt(sum([tf.square(G[:,j] * phi_std[:,j]) for j in range(Npar)]))

    # Create Normal distribution
    dist = tfd.Normal(loc=mean, scale=tf.expand_dims(std, axis=1))
    
    return dist


def inorm_coefficients(par, norm, Npar):
    """
    Unpacks a tensor of shape (N, 6) into mean and standard deviation for each 
    coefficient of the function:

        u(t) = a * cos(ωt) + b * sin(ωt) + c

    Un-normalizes the a, b, and c coefficient means and standard deviations
    """
    # Process the means
    values = []
    scale = 0.5 * norm.denom
    for i in range(Npar - 1):
        norm_mean = tf.expand_dims(par[:, i], axis=1)
        mean = scale * norm_mean
        values.append(mean)
    values.append(scale * (tf.expand_dims(par[:, Npar-1], axis=1) + 1) + norm.xmin)

    # Process the stds
    for i in range(Npar, 2*Npar):
        norm_std = 1.0e-4 + 0.01 * tf.nn.softplus(tf.expand_dims(par[:, i], axis=1))
        std = scale * norm_std
        values.append(std)

    # Return entire list
    return values


class IceStreamNet(tf.keras.Model):

    def __init__(self, bounds, temporal_model=False, name='ice'):

        # Initialize parent class
        super().__init__(name=name)

        # Build the layers
        if temporal_model:
            self.solution = PeriodicSolutionNet(bounds)
        else:  
            self.solution = SolutionNet(bounds)

        # Spacing for finite difference gradients in un-normalized coordinates
        self.delta = tf.constant(0.005, dtype=tf.float32)
        self.x_denom = self.solution.Xn.denom

        # Pre-compute perturbations for 13-point stencil for computing basal drag
        self.dx0 = -2*self.delta
        self.dx1 = -1*self.delta
        self.dx3 =  1*self.delta
        self.dx4 =  2*self.delta

        # Rheology parameters
        self.n = tf.constant(3, dtype=tf.float32)
        self.eta_factor = tf.constant(148365.28, dtype=tf.float32) # T = -5, KPa=False
        self.B_x = tf.constant(-0.001, dtype=tf.float32)

    def call(self, X, T, Xp, Tp, lap_scale=1.0, training=False):

        # Normalized likelihood distributions
        u_dist, h_dist = self.solution(X, T, topo=True)

        # Add extra drag losses
        if Xp is not None:
            self.compute_drag_losses(Xp, Tp, lap_scale=lap_scale)

        return u_dist, h_dist

    def compute_drag_losses(self, X, T, lap_scale=1.0):

        # Compute Laplacian
        lap, tb = self.drag_laplacian(X, T, scale=lap_scale)
        self.add_loss(lap)

        # Sign loss
        sign_loss = 1.0 * tf.reduce_mean(tf.nn.relu(tb))
        self.add_loss(sign_loss)

        return

    def compute_drag_tfgrad(self, X, T, dX=0.0):

        # Physical constants
        rho_ice = 917.0
        rho_water = 1024.0
        g = 9.80665

        # Predict un-normalized mean
        U, H = self.solution.call_inorm(X, T, dX=dX)

        # Compute gradients
        u_x = tf.gradients(U, X)[0]
        h_x = tf.gradients(H, X)[0]

        # Rheology scale factor (n = 3, A = A(T = -5, kPa=True))
        n = 3
        eta_factor = tf.constant(148365.28, dtype=tf.float32)
        strain = tf.abs(u_x)
        eta = eta_factor * strain**((1.0 - n) / n)
        txx = 2.0 * eta * u_x
        tm = 2.0 * tf.gradients(H * txx, X)[0]

        # Compute driving stress
        s_x = h_x + self.B_x
        td = rho_ice * g * H * s_x
   
        # Compute drag
        tb = -1.0e-3 * (tm - td)

        return tb

    def compute_drag(self, Xp, Tp, dX=0.0):
        """
        Computes PDE loss on collocation point predictions.
        """
        # Physical constants
        rho_ice = 917.0
        rho_water = 1024.0
        g = 9.80665

        # Predict (un-normalized) velocity and thickness; discard std
        u0 = self.solution.call_inorm(Xp, Tp, dX=self.dx0, topo=False)
        u1, h1 = self.solution.call_inorm(Xp, Tp, dX=self.dx1, topo=True)
        u2, h2 = self.solution.call_inorm(Xp, Tp, dX=0.0, topo=True)
        u3, h3 = self.solution.call_inorm(Xp, Tp, dX=self.dx3, topo=True)
        u4 = self.solution.call_inorm(Xp, Tp, dX=self.dx4, topo=False)

        # Use velocities to compute stresses
        txx = self.stress(u3, u1)
        txx_fx = self.stress(u4, u2)
        txx_bx = self.stress(u2, u0)

        # Compute stress gradients
        txx_x = self.grad1(txx_fx, txx_bx)

        # Thickness gradients
        h_x = self.grad1(h3, h1)
        s_x = h_x + self.B_x

        # Compute membrane stress
        tm = 2.0 * (h_x * txx + h2 * txx_x)

        # Driving stress
        td = rho_ice * g * h2 * s_x

        # Compute drag (with convention that drag is nominally negative/resistive)
        tb = -1.0e-3 * (tm - td)

        return tb

    def stress(self, u_fx, u_bx):

        # Velocity gradients (in physical, un-normalized coordinates)
        u_x = self.grad1(u_fx, u_bx)

        # Effective strain
        strain = tf.abs(u_x)

        # Effective dynamic viscosity
        eta = self.eta_factor * strain**((1.0 - self.n) / self.n)

        # Stress (kPa)
        txx = 2.0 * eta * u_x
        return txx

    def grad1(self, f, b):
        """
        Convenience function for computing central difference.
        """
        # Central difference w.r.t. normalized coordinate
        d = (f - b) / (2.0 * self.delta)
        # Correct for conversion from normalized -> un-normalized coordinates
        return d / self.x_denom

    def drag_laplacian(self, X, T, scale=1.0):
        """
        Computes un-scaled 1D Laplacian (2nd-order derivative)
        """
        delta = tf.constant(0.01)
        f = self.compute_drag_tfgrad(X, T, dX=delta)
        b = self.compute_drag_tfgrad(X, T, dX=-delta)
        c = self.compute_drag_tfgrad(X, T)
        lap = scale * (f - 2*c + b)
        return tf.reduce_mean(tf.square(lap)), c
        


class SolutionNet(tf.keras.layers.Layer):

    def __init__(self, bounds, name='solution'):
        """
        Build instances of all necessary layers.
        """
        # Initialize parent class
        super().__init__(name=name)

        # Create the layers
        layers = [50, 50, 50, 50, 2]
        init = 'lecun_normal'
        self.u_dense = pgan.networks.DenseNet(layers, initializer=init, name='unet')
        self.h_dense = pgan.networks.DenseNet(layers, initializer=init, name='hnet')

        # Create normalizers for input variables
        self.Xn = pgan.data.Normalizer(*bounds['X'], pos=True)
        self.Tn = pgan.data.Normalizer(*bounds['T'], pos=True)
        # And for output variables
        self.Un = pgan.data.Normalizer(*bounds['U'])
        self.Hn = pgan.data.Normalizer(*bounds['H'])
                
        return

    def call(self, X, T, dX=0.0, topo=True, training=False):
        """
        Takes in a MultiVariable, normalizes it, and passes through dense network.
        """
        # Normalize inputs and add any perturbations
        Xn = self.Xn(X) + dX
        Tn = self.Tn(T)
    
        # Create concatenated tensor
        A = tf.concat(values=[Xn, Tn], axis=1)

        # Send through dense networks to get velocity
        u_par = self.u_dense(A)
        u_dist = unpack_parameters(u_par)

        # Repeat for thickness if requested
        if topo:
            h_par = self.h_dense(A)
            h_dist = unpack_parameters(h_par)
            return u_dist, h_dist

        # Otherwise, return only velocity parameters
        else:
            return u_dist

    def call_inorm(self, X, T, dX=0.0, topo=True, training=False):
        """
        Only used when un-normalized mean is requested.
        """
        pred = self.call(X, T, dX=dX, training=training)
        if topo:
            u_mean = self.Un.inverse(pred[0].loc)
            h_mean = self.Hn.inverse(pred[1].loc)
            return u_mean, h_mean
        else:
            u_mean = self.Un.inverse(pred[0].loc)
            return u_mean


class PeriodicSolutionNet(tf.keras.layers.Layer):

    def __init__(self, bounds, name='solution'):
        """
        Build instances of all necessary layers.
        """
        # Initialize parent class
        super().__init__(name=name)

        # Create the layers
        self.Npar = 6
        layers = [50, 50, 50, 50, 2*self.Npar]
        init = 'lecun_normal'
        self.u_dense = pgan.networks.DenseNet(layers, initializer=init, name='unet')
        self.h_dense = pgan.networks.DenseNet(layers, initializer=init, name='hnet')

        # Create normalizers for input variables
        self.Xn = pgan.data.Normalizer(*bounds['X'], pos=True)
        self.Tn = pgan.data.Normalizer(*bounds['T'], pos=True)
        # And for output variables
        self.Un = pgan.data.Normalizer(*bounds['U'])
        self.Hn = pgan.data.Normalizer(*bounds['H'])

        return

    def call(self, X, G, dX=0.0, topo=True, training=False):
        """
        Takes in a MultiVariable, normalizes it, and passes through dense network.
        """
        # Normalize inputs and add any perturbations
        Xn = self.Xn(X) + dX

        # Send through dense networks to get coefficient mean/std
        u_par = self.u_dense(Xn)
        # Create distribution
        u_dist = unpack_coeff_parameters(u_par, G, self.Npar)

        # Repeat for thickness if requested
        if topo:
            h_par = self.h_dense(Xn)
            h_dist = unpack_coeff_parameters(h_par, G, self.Npar)
            return u_dist, h_dist

        # Otherwise, return only velocity distribution
        else:
            return u_dist

    def call_inorm(self, X, G, dX=0.0, topo=True, training=False):
        """
        Only used when un-normalized mean is requested.
        """
        pred = self.call(X, G, dX=dX, training=training)
        if topo:
            u_mean = self.Un.inverse(pred[0].loc)
            h_mean = self.Hn.inverse(pred[1].loc)
            return u_mean, h_mean
        else:
            u_mean = self.Un.inverse(pred[0].loc)
            return u_mean

    def call_inorm_coefficients(self, X, training=False):
        """
        Computes coefficients of periodic function and un-normalizes them.
        """
        # Normalize inputs
        Xn = self.Xn(X)

        # Send through networks to get coefficients
        u_par = self.u_dense(Xn)
        h_par = self.h_dense(Xn)

        # Un-normalize
        u_coeffs = inorm_coefficients(u_par, self.Un, self.Npar)
        h_coeffs = inorm_coefficients(h_par, self.Hn, self.Npar)

        return u_coeffs, h_coeffs


# end of file
