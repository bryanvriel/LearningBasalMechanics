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

# Processing parameters
USE_TEMPORAL = True # set to True to use temporal design matrix G
RESTORE = True
BATCH_SIZE = 512
LEARNING_RATE = 0.0003
N_EPOCHS = 500

def main():

    # Initialize logging data
    logging.basicConfig(filename='log_pretrain', filemode='w', level=logging.INFO)

    # Load data and bounds
    data_fit, _, bounds = load_data(seed=13, batch_size=BATCH_SIZE, loadG=USE_TEMPORAL)

    # --------------------------------------------------------------------------------
    # Create networks, optimizer, and losses
    # --------------------------------------------------------------------------------

    # Create model
    model = networks.IceStreamNet(bounds, temporal_model=USE_TEMPORAL)

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    @tf.function
    def compute_losses(X, T, U, H):
        
        # Predict likelihoods
        u_dist, h_dist = model(X, T, None, None)
        
        # Compute negative log-likelihood
        u_nll = -2.0 * tf.reduce_mean(u_dist.log_prob(U))
        h_nll = -2.0 * tf.reduce_mean(h_dist.log_prob(H))

        return u_nll, h_nll

    # JIT the update function
    b = data_fit.train_batch()
    values = compute_losses(b.X, b.T, b.U, b.H)

    @tf.function
    def update(X, T, U, H):

        # Compute gradient of total loss
        with tf.GradientTape() as tape:
            u_nll, h_nll = compute_losses(X, T, U, H)
            total_loss = u_nll + h_nll
        grads = tape.gradient(total_loss, model.trainable_variables)

        # Apply gradients
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Return losses
        return u_nll, h_nll

    # JIT the update function
    b = data_fit.train_batch()
    values = update(b.X, b.T, b.U, b.H)
    n_loss = len(values)

    # Reset training counters for data objects
    data_fit.reset_training()

    # Create checkpoint objects for variables
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(ckpt, 'checkpoints_pretrain', 1)

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

            # Update
            values = update(b.X, b.T, b.U, b.H)
            train_vals[bnum, :] = [v.numpy() for v in values]

        # Reset training counters for data objects
        data_fit.reset_training()

        # Mean train loss
        train = np.mean(train_vals, axis=0).tolist()

        # Compute losses on test batches
        b = data_fit.test_batch()
        values = compute_losses(b.X, b.T, b.U, b.H)
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
