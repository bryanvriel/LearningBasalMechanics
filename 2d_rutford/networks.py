#-*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import pgan
import sys


class IceStreamNet(tf.keras.Model):

    def __init__(self, bounds, name='ice'):

        # Initialize parent class
        super().__init__(name=name)

        # Build the layers
        self.solution = PeriodicSolutionNet(bounds)

        # Spacing for finite difference gradients in un-normalized coordinates
        δ = tf.constant(0.00375, dtype=tf.float32)
        self.δ = δ
        self.x_denom = self.solution.Xn.denom
        self.y_denom = self.solution.Yn.denom

        # Pre-compute perturbations for 13-point stencil for computing basal drag
        self.δx02, self.δy02 = 0, 2*δ
        self.δx11, self.δy11 = -δ, δ
        self.δx12, self.δy12 =  0, δ
        self.δx13, self.δy13 = δ, δ
        self.δx20, self.δy20 = -2*δ, 0
        self.δx21, self.δy21 =  -δ, 0
        self.δx22, self.δy22 = 0, 0
        self.δx23, self.δy23 =  δ, 0
        self.δx24, self.δy24 = 2*δ, 0
        self.δx31, self.δy31 = -δ, -δ
        self.δx32, self.δy32 =  0, -δ
        self.δx33, self.δy33 = δ, -δ
        self.δx42, self.δy42 = 0, -2*δ

        # Rheology parameters (eta_factor = 0.5 * A**(-1 / n))
        self.n = tf.constant(3, dtype=tf.float32)
        self.eta_factor = tf.constant(221.75151, dtype=tf.float32) # T = -10, KPa=True

    def call(self, X, Y, T, Xp, Yp, Tp, lap_scale=1.0, training=False):

        # Normalized likelihood distributions
        u_dist, v_dist, h_dist, s_dist = self.solution(X, Y, T, topo=True)

        # Add extra drag losses
        if Xp is not None:
            self.compute_drag_losses(Xp, Yp, Tp, lap_scale=lap_scale)

        return u_dist, v_dist, h_dist, s_dist

    def compute_drag_losses(self, X, Y, T, lap_scale=1.0):

        # Compute Laplacian
        lap, tb = self.drag_laplacian(X, Y, T, scale=lap_scale)
        self.add_loss(lap)

        # Sign loss
        sign_loss = 1.0 * tf.reduce_mean(tf.nn.relu(tb))
        self.add_loss(sign_loss)

        return

    def compute_drag_tfgrad(self, X, Y, G, dX=0.0, dY=0.0):

        # Physical constants
        rho_ice = tf.constant(917.0, dtype=tf.float32)
        rho_water = tf.constant(1024.0, dtype=tf.float32)
        g = tf.constant(9.80665, dtype=tf.float32)

        # Predict un-normalized mean
        U, V, H, S = self.solution.call_inorm(X, Y, G, dX=dX, dY=dY)

        # Compute gradients
        u_x = tf.gradients(U, X)[0]
        v_x = tf.gradients(V, X)[0]
        u_y = tf.gradients(U, Y)[0]
        v_y = tf.gradients(V, Y)[0]
        s_x = tf.gradients(S, X)[0]
        s_y = tf.gradients(S, Y)[0]

        # Effective viscosity
        strain = tf.sqrt(u_x**2 + v_y**2 + 0.25 * (u_y + v_x)**2 + u_x * v_y)
        eta = self.eta_factor * strain**((1.0 - self.n) / self.n)

        # Membrane stresses in X-direction
        tmxx = tf.gradients(2.0 * eta * H * (2.0 * u_x + v_y), X)[0]
        tmxy = tf.gradients(eta * H * (u_y + v_x), Y)[0]

        # Membrane stresses in Y-direction
        tmyy = tf.gradients(2.0 * eta * H * (2.0 * v_y + u_x), Y)[0]
        tmyx = tf.gradients(eta * H * (u_y + v_x), X)[0]

        # Driving stresses (convert to kPa to be consistent with membrane)
        tdx = 1.0e-3 * rho_ice * g * H * s_x
        tdy = 1.0e-3 * rho_ice * g * H * s_y

        # Compute drag components
        tbx = -1.0 * (tmxx + tmxy - tdx)
        tby = -1.0 * (tmyy + tmyx - tdy)

        # Compute along-flow
        mag = tf.sqrt(tf.square(U) + tf.square(V))
        tb = tbx * U / mag + tby * V / mag
 
        return tb

    def compute_drag(self, X, Y, T, dX=0.0, dY=0.0):
        """
        Computes PDE loss on collocation point predictions.
        """
        # Physical constants
        rho_ice = 917.0
        rho_water = 1024.0
        g = 9.80665

        # Need to compute velocities at 13 different stencil points and
        # topography at 4 stencil points
        u02, v02 = self.solution.call_inorm(X, Y, T, topo=False, dX=self.δx02+dX, dY=self.δy02+dY)
        u11, v11 = self.solution.call_inorm(X, Y, T, topo=False, dX=self.δx11+dX, dY=self.δy11+dY)
        u13, v13 = self.solution.call_inorm(X, Y, T, topo=False, dX=self.δx13+dX, dY=self.δy13+dY)
        u20, v20 = self.solution.call_inorm(X, Y, T, topo=False, dX=self.δx20+dX, dY=self.δy20+dY)
        u24, v24 = self.solution.call_inorm(X, Y, T, topo=False, dX=self.δx24+dX, dY=self.δy24+dY)
        u31, v31 = self.solution.call_inorm(X, Y, T, topo=False, dX=self.δx31+dX, dY=self.δy31+dY)
        u33, v33 = self.solution.call_inorm(X, Y, T, topo=False, dX=self.δx33+dX, dY=self.δy33+dY)
        u42, v42 = self.solution.call_inorm(X, Y, T, topo=False, dX=self.δx42+dX, dY=self.δy42+dY)

        u12, v12, h12, s12 = self.solution.call_inorm(X, Y, T, dX=self.δx12+dX, dY=self.δy12+dY)
        u21, v21, h21, s21 = self.solution.call_inorm(X, Y, T, dX=self.δx21+dX, dY=self.δy21+dY)
        u22, v22, h22, s22 = self.solution.call_inorm(X, Y, T, dX=self.δx22+dX, dY=self.δy22+dY)
        u23, v23, h23, s23 = self.solution.call_inorm(X, Y, T, dX=self.δx23+dX, dY=self.δy23+dY)
        u32, v32, h32, s32 = self.solution.call_inorm(X, Y, T, dX=self.δx32+dX, dY=self.δy32+dY)

        # Use velocities to compute stresses at 5 stencil points
        txx, tyy, txy = self.stress(u23, u21, u12, u32, v23, v21, v12, v32)
        txx_fx, tyy_fx, txy_fx = self.stress(u24, u22, u13, u33, v24, v22, v13, v33)
        txx_bx, tyy_bx, txy_bx = self.stress(u22, u20, u11, u31, v22, v20, v11, v31)
        txx_fy, tyy_fy, txy_fy = self.stress(u13, u11, u02, u22, v13, v11, v02, v22)
        txx_by, tyy_by, txy_by = self.stress(u33, u31, u22, u42, v33, v31, v22, v42)

        # Thickness gradients
        h_x = self.gradx(h23, h21)
        h_y = self.grady(h12, h32)
        s_x = self.gradx(s23, s21)
        s_y = self.grady(s12, s32)

        # Compute membrane stress components
        tmxx = h_x * (2 * txx + tyy) + \
               h22 * (2 * self.gradx(txx_fx, txx_bx) + self.gradx(tyy_fx, tyy_bx))
        tmxy = h_y * txy + h22 * self.grady(txy_fy, txy_by)

        tmyy = h_y * (2 * tyy + txx) + \
               h22 * (2 * self.grady(tyy_fy, tyy_by) + self.grady(txx_fy, txx_by))
        tmyx = h_x * txy + h22 * self.gradx(txy_fx, txy_bx)

        # Compute driving stress
        tdx = 1.0e-3 * rho_ice * g * h22 * s_x
        tdy = 1.0e-3 * rho_ice * g * h22 * s_y

        # Compute drag components
        tbx = -1.0 * (tmxx + tmxy - tdx)
        tby = -1.0 * (tmyy + tmyx - tdy)

        # Project along-flow
        mag = tf.sqrt(tf.square(u22) + tf.square(v22))
        tb = tbx * u22 / mag + tby * v22 / mag

        return tb

    def stress(self, u_fx, u_bx, u_fy, u_by, v_fx, v_bx, v_fy, v_by):

        # Velocity gradients (in physical, un-normalized coordinates)
        u_x = self.gradx(u_fx, u_bx)
        v_x = self.gradx(v_fx, v_bx)
        v_y = self.grady(v_fy, v_by)
        u_y = self.grady(u_fy, u_by)

        # Effective strain
        strain = tf.sqrt(u_x**2 + v_y**2 + 0.25 * (u_y + v_x)**2 + u_x * v_y)

        # Effective dynamic viscosity
        eta = self.eta_factor * strain**((1.0 - self.n) / self.n)

        # Stress components
        txx = 2.0 * eta * u_x
        tyy = 2.0 * eta * v_y
        txy = eta * (u_y + v_x)

        return txx, tyy, txy

    def gradx(self, f, b):
        """
        Convenience function for computing central difference.
        """
        # Central difference w.r.t. normalized coordinate
        d = (f - b) / (2.0 * self.δ)
        # Correct for conversion from normalized -> un-normalized coordinates
        return d / self.x_denom

    def grady(self, f, b):
        """
        Convenience function for computing central difference.
        """
        # Central difference w.r.t. normalized coordinate
        d = (f - b) / (2.0 * self.δ)
        # Correct for conversion from normalized -> un-normalized coordinates
        return d / self.y_denom

    def drag_laplacian(self, X, Y, G, scale=1.0):
        """
        Computes un-scaled 1D Laplacian (2nd-order derivative)
        """
        #delta = tf.constant(0.00375, dtype=tf.float32)
        delta = tf.constant(0.01, dtype=tf.float32)

        # Predict center
        z = self.compute_drag(X, Y, G)

        # Predict ahead
        zfx = self.compute_drag(X, Y, G, dX=delta)
        zfy = self.compute_drag(X, Y, G, dY=delta)

        # Predict behind
        zbx = self.compute_drag(X, Y, G, dX=-delta)
        zby = self.compute_drag(X, Y, G, dY=-delta)

        # Compute Laplacian
        dx2 = scale * (zfx - 2.0 * z + zbx)
        dy2 = scale * (zfy - 2.0 * z + zby)
        return tf.reduce_mean(tf.square(dx2 + dy2)), z

    def drag_total_variation(self, X, Y, G, scale=1.0):
        """
        Computes un-scaled 2D total variation (1nd-order derivative). Alternative
        to Laplacian smoothing.
        """
        # FD spacing
        delta = tf.constant(0.00375, dtype=tf.float32)

        # Predict center
        z = self.compute_drag(X, Y, G)

        # Predict ahead
        zfx = self.compute_drag(X, Y, G, dX=delta)
        zfy = self.compute_drag(X, Y, G, dY=delta)

        # Predict behind
        zbx = self.compute_drag(X, Y, G, dX=-delta)
        zby = self.compute_drag(X, Y, G, dY=-delta)

        # Compute total variation
        dx = scale * (zfx - zbx)
        dy = scale * (zfy - zby)
        TV = tf.reduce_mean(tf.sqrt(tf.square(dx) + tf.square(dy)))
        return TV, z
        

class PeriodicSolutionNet(tf.keras.layers.Layer):

    def __init__(self, bounds, name='solution'):
        """
        Build instances of all necessary layers.
        """
        # Initialize parent class
        super().__init__(name=name)

        self.Npar = 3
        u_layers = [50, 50, 50, 50, 50, 50, 2*2*self.Npar]
        h_layers = [50, 50, 50, 50, 4]
        init = 'lecun_normal'

        # Create the layers: one network for 2 velocity components, another network for
        # 2 (time-invariant) topo components
        self.u_dense = pgan.networks.DenseNet(u_layers, initializer=init, name='unet')
        self.h_dense = pgan.networks.DenseNet(h_layers, initializer=init, name='hnet')

        # Create normalizers for input variables
        self.Xn = pgan.data.Normalizer(*bounds['X'], pos=True)
        self.Yn = pgan.data.Normalizer(*bounds['Y'], pos=True)
        # And for output variables
        self.Un = pgan.data.Normalizer(*bounds['U'])
        self.Vn = pgan.data.Normalizer(*bounds['V'])
        self.Hn = pgan.data.Normalizer(*bounds['H'])
        self.Sn = pgan.data.Normalizer(*bounds['S'])

        return

    def call(self, X, Y, G, dX=0.0, dY=0.0, topo=True, training=False):
        """
        Takes in a MultiVariable, normalizes it, and passes through dense network.
        """
        # Normalize inputs and add any perturbations
        Xn = self.Xn(X) + dX
        Yn = self.Yn(Y) + dY

        # Create concatenated tensor for spatial coordinates
        An = tf.concat(values=[Xn, Yn], axis=1)

        # Send through dense networks to get coefficient mean/std
        u_par = self.u_dense(An)
        # Create distributions for two velocity components
        u_dist, v_dist = unpack_coeff_parameters(u_par, G, self.Npar)

        # Repeat for thickness if requested
        if topo:
            h_par = self.h_dense(An)
            h_dist, s_dist = unpack_parameters(h_par)
            return u_dist, v_dist, h_dist, s_dist

        # Otherwise, return only velocity distributions
        else:
            return u_dist, v_dist

    def call_inorm(self, X, Y, G, dX=0.0, dY=0.0, topo=True, training=False):
        """
        Only used when un-normalized means are requested.
        """
        pred = self.call(X, Y, G, dX=dX, dY=dY, training=training)
        u = self.Un.inverse(pred[0].loc)
        v = self.Vn.inverse(pred[1].loc)
        if topo:
            h = self.Hn.inverse(pred[2].loc)
            s = self.Sn.inverse(pred[3].loc)
            return u, v, h, s
        else:
            return u, v

    def call_inorm_full(self, X, Y, G, dX=0.0, dY=0.0, topo=True, training=False):
        """
        Only used when un-normalized means are requested.
        """
        pred = self.call(X, Y, G, dX=dX, dY=dY, training=training)
        u = self.Un.inverse(pred[0].loc)
        v = self.Vn.inverse(pred[1].loc)
        u_std = 0.5 * self.Un.denom * pred[0].scale
        v_std = 0.5 * self.Vn.denom * pred[1].scale
        if topo:
            h = self.Hn.inverse(pred[2].loc)
            s = self.Sn.inverse(pred[3].loc)
            h_std = 0.5 * self.Hn.denom * pred[2].scale
            s_std = 0.5 * self.Sn.denom * pred[3].scale
            return u, v, h, s, u_std, v_std, h_std, s_std
        else:
            return u, v, u_std, v_std

    def call_inorm_coefficients(self, X, training=False):
        """
        Computes coefficients of periodic function and un-normalizes them.
        """
        raise NotImplementedError('Cannot unpack coefficients yet.')        

        # Normalize inputs
        Xn = self.Xn(X)

        # Send through networks to get coefficients
        u_par = self.u_dense(Xn)
        h_par = self.h_dense(Xn)

        # Un-normalize
        u_coeffs = inorm_coefficients(u_par, self.Un, self.Npar)
        h_coeffs = inorm_coefficients(h_par, self.Hn, self.Npar)

        return u_coeffs, h_coeffs


# --------------------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------------------


def unpack_parameters(par):
    """
    Unpacks a tensor of shape (N, 4) into means and standard deviations. Creates
    two separate probability distributions.
    """
    μ1 = tf.expand_dims(par[:, 0], axis=1)
    μ2 = tf.expand_dims(par[:, 1], axis=1)
    σ1 = 1.0e-4 + 0.01 * tf.nn.softplus(tf.expand_dims(par[:, 2], axis=1))
    σ2 = 1.0e-4 + 0.01 * tf.nn.softplus(tf.expand_dims(par[:, 3], axis=1))
    dist1 = tfd.Normal(loc=μ1, scale=σ1)
    dist2 = tfd.Normal(loc=μ2, scale=σ2)

    return dist1, dist2


def unpack_coeff_parameters(par, G, Npar):
    """
    Unpacks a tensor of shape (N, 2*2*Npar), where Npar = 3, into mean and standard deviation
    for each coefficient of the function:

        u(t) = c + a * cos(ωt) + b * sin(ωt)

    Return only the ::predicted:: mean and standard deviation using
    propagation of uncertainties.
    """
    # Partition the means and standard deviations
    phi1_mean = par[:, :Npar]
    phi2_mean = par[:, Npar:2*Npar]
    phi1_std = 1.0e-4 + 0.01 * tf.nn.softplus(par[:, 2*Npar:3*Npar])
    phi2_std = 1.0e-4 + 0.01 * tf.nn.softplus(par[:, 3*Npar:])

    # Multiply by design matrix
    μ1 = tf.reduce_sum(G * phi1_mean, axis=1, keepdims=True)
    μ2 = tf.reduce_sum(G * phi2_mean, axis=1, keepdims=True)

    # Compute standard deviation
    σ1 = tf.sqrt(sum([tf.square(G[:,j] * phi1_std[:,j]) for j in range(Npar)]))
    σ2 = tf.sqrt(sum([tf.square(G[:,j] * phi2_std[:,j]) for j in range(Npar)]))

    # Create Normal distributions
    dist1 = tfd.Normal(loc=μ1, scale=tf.expand_dims(σ1, axis=1))
    dist2 = tfd.Normal(loc=μ2, scale=tf.expand_dims(σ2, axis=1))

    return dist1, dist2


def inorm_velocity_coefficients(par, norm, Npar):
    """
    Unpacks a tensor of shape (N, 6) into mean and standard deviation for each 
    coefficient of the function:

        u(t) = a * cos(ωt) + b * sin(ωt) + c

    Un-normalizes the a, b, and c coefficient means and standard deviations
    """
    raise NotImplementedError
    u_values = []; v_values = []
    scale = 0.5 * norm.denom

    # Process the means
    values = []
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


# end of file
