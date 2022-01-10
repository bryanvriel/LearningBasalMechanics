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

# Locals
import networks
from utilities import *
from pretrain import USE_TEMPORAL

# Processing parameters
#RESTORE = False; LOAD_PRETRAIN = True
RESTORE = True; LOAD_PRETRAIN = False
BATCH_SIZE = 512
LEARNING_RATE = 0.0003
N_EPOCHS = 500


def main():

    # Initialize logging data
    logging.basicConfig(filename='log_train', filemode='w', level=logging.INFO)

    # Load data and bounds
    data_fit, data_pde, bounds = load_data(seed=13, batch_size=BATCH_SIZE, loadG=USE_TEMPORAL)

    # --------------------------------------------------------------------------------
    # Create networks, optimizer, and losses
    # --------------------------------------------------------------------------------

    # Create model
    model = networks.IceStreamNet(bounds, temporal_model=USE_TEMPORAL)

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    @tf.function
    def compute_losses(X, T, Xp, Tp, U, H):
        
        # Predict likelihoods
        u_dist, h_dist = model(X, T, Xp, Tp, lap_scale=5.0)
        
        # Compute negative log-likelihood
        u_nll = -2.0 * tf.reduce_mean(u_dist.log_prob(U))
        h_nll = -2.0 * tf.reduce_mean(h_dist.log_prob(H))

        # Combine with physics losses
        losses = [u_nll, h_nll] + model.losses

        return losses

    @tf.function
    def update(X, T, Xp, Tp, U, H):

        # Compute gradient of total loss
        with tf.GradientTape() as tape:
            losses = compute_losses(X, T, Xp, Tp, U, H)
            total_loss = sum(losses)
        grads = tape.gradient(total_loss, model.trainable_variables)

        # Clip gradients?
        #grads, _ = tf.clip_by_global_norm(grads, 4.0)

        # Apply gradients
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Return losses
        return losses

    # JIT the update function
    b = data_fit.train_batch()
    bp = data_pde.train_batch()
    values = update(b.X, b.T, bp.X, bp.T, b.U, b.H)
    n_loss = len(values)

    # Reset training counters for data objects
    data_fit.reset_training()
    data_pde.reset_training()

    # Checkpoint and manager for pretrained weights
    load_pretrain = LOAD_PRETRAIN
    if load_pretrain:
        pre_ckpt = tf.train.Checkpoint(model=model)
        pre_manager = tf.train.CheckpointManager(pre_ckpt, 'checkpoints_pretrain', 1)
        pre_ckpt.restore(pre_manager.latest_checkpoint).expect_partial()

    # Create checkpoint objects for variables
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(ckpt, 'checkpoints', 1)

    # Restore
    if RESTORE:
        ckpt.restore(manager.latest_checkpoint)

    # --------------------------------------------------------------------------------
    # Training
    # --------------------------------------------------------------------------------

    # Training iterations
    for epoch in tqdm(range(N_EPOCHS), desc='Epoch loop'):

        # Loop over batches
        train_vals = np.zeros((data_fit.n_batches, n_loss))
        for bnum in range(data_fit.n_batches):

            # Get batches
            b = data_fit.train_batch()
            bp = data_pde.train_batch()

            # Update
            values = update(b.X, b.T, bp.X, bp.T, b.U, b.H)
            train_vals[bnum, :] = [v.numpy() for v in values]

        # Reset training counters for data objects
        data_fit.reset_training()
        data_pde.reset_training()

        # Mean train loss
        train = np.mean(train_vals, axis=0).tolist()

        # Compute losses on test batches
        b = data_fit.test_batch()
        bp = data_pde.test_batch()
        values = compute_losses(b.X, b.T, bp.X, bp.T, b.U, b.H)
        test = [v.numpy() for v in values]

        # Write to logfile
        out = '%d ' + '%f ' * 2 * n_loss
        logging.info(out % tuple([epoch] + train + test))

        # Save temporary checkpoints
        if epoch % 20 == 0 and epoch > 0:
            manager.save()

    # Save final weights
    manager.save()


if __name__ == '__main__':
    main()

# end of file
