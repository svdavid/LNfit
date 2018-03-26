import numpy as np
import tensorflow as tf
import scipy.io as sio
import scipy.signal as sig
import h5py
import os
import pickle
import sys
import importlib

def weight_variable(shape, sig=0.01):
    initial = tf.truncated_normal(shape, stddev=sig)
    return tf.Variable(initial)


def bias_variable(shape, val=0.01):
    initial = tf.constant(val, shape=shape)
    return tf.Variable(initial)


def conv1d(k, Y):
    # convolution function
    # k: batch x time x feature
    # W: time x feature x output channel
    # returns: batch x time x output channel

    return tf.nn.conv1d(k, Y, stride=1, padding='SAME')

def onelayer(F, D, batch_size, train_test_folds, n_iter, step_size, rank=None, time_win=25, 
             layer1_scale=0.01, nonlin='relu', print_iterations=True,
             l2_penalty=0, directory_to_save=None, fname_flag='LN', overwrite=False,
             fix_bias=None, early_stopping=True, optimizer='adam', momentum=0.9, use_nesterov=False):

    # create directroy if not present
    if not(os.path.exists(directory_to_save)):
        os.makedirs(directory_to_save)
    
    idstring = ('iter' + str(n_iter) + '-timewin' + str(time_win) + '-step' + str(step_size)
                 + '-scale' + str(layer1_scale) + '-nonlin-'  + str(nonlin) + '-l2penalty' + str(l2_penalty)
                 + '-rank' + str(rank) + '-earlystop' + str(early_stopping))
    
    if not(fix_bias is None):
        idstring = (idstring + '-bias' + str(fix_bias))

    if not(optimizer=='adam'):
        idstring = (idstring + '-' + optimizer)
        if optimizer=='momentum':
            idstring = (idstring + str(momentum) + '-nest' + str(use_nesterov))

    if directory_to_save is None:
        fname = ''
    else:
        fname = (directory_to_save + '/' + fname_flag + '-' + idstring + '.p')
        
        if os.path.exists(fname) and not(overwrite):
            Z = pickle.load(open(fname, "rb"))
            return Z
        
    # separate out train and testing
    D_train_optimize = D[train_test_folds == 0, :, :]
    D_train_crossval = D[train_test_folds == 1, :, :]
    D_test = D[train_test_folds == 2, :, :]
    F_train_optimize = F[train_test_folds == 0, :, :]
    F_train_crossval = F[train_test_folds == 1, :, :]
    F_test = F[train_test_folds == 2, :, :]

    n_timepoints = F_train_optimize.shape[1]
    n_features = F_train_optimize.shape[2]
    n_electrodes = D_train_optimize.shape[2]

    # scale of weights and biases for layer 1
    num_inputs = time_win * n_features
    scaling_factor = 1 / np.sqrt(num_inputs)
    scale = np.float32(layer1_scale * scaling_factor *
                       np.std(D_train_optimize) / np.std(F_train_optimize))
    
    n_opt_stim = D_train_optimize.shape[0]
    
    train_order = np.random.permutation(n_opt_stim)
        
    with tf.Session() as sess:

        # place-holder for features and data
        f = tf.placeholder(tf.float32, shape=[None, n_timepoints, n_features])
        data = tf.placeholder(tf.float32, shape=[None, n_timepoints, n_electrodes])

        # rank-constrained weights
        if rank is None:
            W = weight_variable([time_win, n_features, n_electrodes], sig=layer1_scale)
        else:
            Wt = []
            Wf = []
            for i in range(n_electrodes):
                Wt.append(weight_variable([time_win, rank], sig=layer1_scale))
                Wf.append(weight_variable([rank, n_features], sig=layer1_scale))
            Wtf = []
            for i in range(n_electrodes):
                Wtf.append(tf.matmul(Wt[i], Wf[i]))
            W = tf.reshape(tf.concat(Wtf, 2), [time_win, n_features, n_electrodes])
        
        # bias
        if fix_bias is None:
            b = bias_variable([n_electrodes], val=scale)
        else:
            b = tf.constant(fix_bias, shape=[n_electrodes], dtype=tf.float32)
        
        # output layer
        if nonlin == 'relu':
            h = tf.nn.relu(conv1d(f, W) + b)
        elif nonlin == 'tanh':
            h = tf.nn.tanh(conv1d(f, W) + b)
        elif nonlin == 'none':
            h = conv1d(f, W) + b
        else:
            raise NameError('no matching nonlin')

        # reshape to format of the data
        prediction = tf.reshape(h, [-1, n_timepoints, n_electrodes])

        # squared error loss
        loss = tf.reduce_mean(tf.square(data - prediction))

        # augment with penalties
        penalized_loss = loss + l2_penalty * tf.reduce_mean(tf.square(W))

        # gradient optimizer
        if optimizer=='adam':
            train_step = tf.train.AdamOptimizer(step_size).minimize(penalized_loss)
        elif optimizer=='sgd':
            train_step = tf.train.GradientDescentOptimizer(step_size).minimize(penalized_loss)
        elif optimizer=='momentum':
            train_step = tf.train.MomentumOptimizer(step_size, momentum, use_nesterov=use_nesterov).minimize(penalized_loss)
        else:
            raise NameError('No matching optimizer')

        # initialize global variables
        sess.run(tf.global_variables_initializer())

        # initialize and train
        penalized_train_loss = np.zeros((n_iter, 1))
        train_loss = np.zeros((n_iter, 1))
        test_loss = np.zeros((n_iter, 1))
        smallest_test_loss = loss.eval(
            feed_dict={f: F_train_crossval, data: D_train_crossval})
        W_best_value = W.eval()
        b_best_value = b.eval()
        batch_index = np.arange(0, batch_size)
        for i in range(n_iter):
            
            bi = train_order[np.mod(batch_index, n_opt_stim)]
            
            if print_iterations and np.mod(i, np.round(n_iter / 20)) == 0:
                print(i)
            train_step.run(feed_dict={f: F_train_optimize[bi, :, :], 
                                      data: D_train_optimize[bi, :, :]})
            penalized_train_loss[i] = penalized_loss.eval(
                feed_dict={f: F_train_optimize[bi, :, :], 
                           data: D_train_optimize[bi, :, :]})
            train_loss[i] = loss.eval(
                feed_dict={f: F_train_optimize[bi, :, :], 
                           data: D_train_optimize[bi, :, :]})
            test_loss[i] = loss.eval(
                feed_dict={f: F_train_crossval, data: D_train_crossval})
            if early_stopping and (test_loss[i] < smallest_test_loss):
                smallest_test_loss = test_loss[i]
                W_best_value = W.eval()
                b_best_value = b.eval()

            batch_index = batch_index + batch_size
            if batch_index[0] > n_opt_stim:
                batch_index = np.arange(0, batch_size)
                train_order = np.random.permutation(np.random.permutation(n_opt_stim))

        if not(early_stopping):
            W_best_value = W.eval()
            b_best_value = b.eval()

        prediction_fn = lambda F_value: prediction.eval(
            feed_dict={f: F_value, W: W_best_value, b: b_best_value})

        P = prediction_fn(F)

        Z = {}
        Z['P'] = P
        Z['test_loss'] = test_loss
        Z['train_loss'] = train_loss
        Z['W'] = W_best_value
        Z['b'] = b_best_value
        Z['penalized_train_loss'] = penalized_train_loss
        Z['fname'] = fname
        Z['idstring'] = idstring

        if not(directory_to_save is None):
            pickle.dump(Z, open(fname, "wb"))
    
    return Z