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

def main():

    # Initialize logging data
    logging.basicConfig(filename='log_pretrain', filemode='w', level=logging.INFO)

    # Load data and bounds
    data_fit, _, bounds = load_data(seed=40, batch_size=1024)

    # --------------------------------------------------------------------------------
    # Create networks, optimizer, and losses
    # --------------------------------------------------------------------------------

    # Create model
    model = networks.IceStreamNet(bounds)

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    @tf.function
    def compute_losses(X, Y, T, U, V, H, S):
        
        # Predict likelihoods
        u_dist, v_dist, h_dist, s_dist = model(X, Y, T, None, None, None)
        
        # Compute negative log-likelihoods
        u_nll = -2.0 * tf.reduce_mean(u_dist.log_prob(U))
        v_nll = -2.0 * tf.reduce_mean(v_dist.log_prob(V))
        h_nll = -2.0 * tf.reduce_mean(h_dist.log_prob(H))
        s_nll = -2.0 * tf.reduce_mean(s_dist.log_prob(S))
        losses = [u_nll, v_nll, h_nll, s_nll]

        return losses

    # JIT the update function
    b = data_fit.train_batch()
    values = compute_losses(b.X, b.Y, b.T, b.U, b.V, b.H, b.S)

    @tf.function
    def update(X, Y, T, U, V, H, S):

        # Compute gradient of total loss
        with tf.GradientTape() as tape:
            losses = compute_losses(X, Y, T, U, V, H, S)
            total_loss = sum(losses)
        grads = tape.gradient(total_loss, model.trainable_variables)

        # Clip gradients?
        grads, _ = tf.clip_by_global_norm(grads, 5.0)

        # Apply gradients
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Return losses
        return losses

    # JIT the update function
    b = data_fit.train_batch()
    values = update(b.X, b.Y, b.T, b.U, b.V, b.H, b.S)
    n_loss = len(values)

    # Reset training counters for data objects
    data_fit.reset_training()

    # Create checkpoint objects for variables
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(ckpt, 'checkpoints_pretrain', 1)

    # Restore
    restore = False
    if restore:
        ckpt.restore(manager.latest_checkpoint)
        n_epochs = 1000
    else:
        n_epochs = 500

    # --------------------------------------------------------------------------------
    # Training
    # --------------------------------------------------------------------------------

    # Training iterations
    for epoch in tqdm(range(n_epochs), desc='Epoch loop'):

        # Loop over batches
        train_vals = np.zeros((data_fit.n_batches, n_loss))
        for bnum in range(data_fit.n_batches):

            # Get batches
            b = data_fit.train_batch()

            # Update
            values = update(b.X, b.Y, b.T, b.U, b.V, b.H, b.S)
            train_vals[bnum, :] = [v.numpy() for v in values]

        # Reset training counters for data objects
        data_fit.reset_training()

        # Mean train loss
        train = np.mean(train_vals, axis=0).tolist()

        # Compute losses on test batches
        b = data_fit.test_batch()
        values = compute_losses(b.X, b.Y, b.T, b.U, b.V, b.H, b.S)
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
