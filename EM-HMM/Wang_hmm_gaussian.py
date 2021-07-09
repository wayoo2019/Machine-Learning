#!/usr/bin/env python3
import numpy as np
if not __file__.endswith('_hmm_gaussian.py'):
    print('ERROR: This file is not named correctly! Please name it as Lastname_hmm_gaussian.py (replacing Lastname with your last name)!')
    exit(1)

DATA_PATH = "/Users/wayoo/446project/hmm_smoke_test_files"  #TODO: if doing development somewhere other than the cycle server (not recommended), then change this to the directory where your data file is (points.dat)

def parse_data(args):
    num = float
    dtype = np.float32
    data = []
    with open(args.data_file, 'r') as f:
        for line in f:
            data.append([num(t) for t in line.split()])
    dev_cutoff = int(.9*len(data))
    train_xs = np.asarray(data[:dev_cutoff],dtype=dtype)
    dev_xs = np.asarray(data[dev_cutoff:],dtype=dtype) if not args.nodev else None
    return train_xs, dev_xs

def init_model(args):
    if args.cluster_num:
        mus = np.zeros((args.cluster_num,2))
        if not args.tied:
            sigmas = np.zeros((args.cluster_num,2,2))
        else:
            sigmas = np.zeros((2,2))
        transitions = np.zeros((args.cluster_num,args.cluster_num)) #transitions[i][j] = probability of moving from cluster i to cluster j
        initials = np.zeros(args.cluster_num) #probability for starting in each state
        #TODO: randomly initialize clusters (mus, sigmas, initials, and transitions)
        K = args.cluster_num
        np.random.seed(0)
        mus = np.random.uniform(0.0, 1.0, (mus.shape))
        if args.tied:
            sigmas = np.eye(2)
        else:
            for k in range(K):
                sigmas[k] = np.eye(2)
        initials = np.tile(1/K, (K))
        transitions = np.random.rand(K, K)
        transitions = transitions/transitions.sum(axis=1, keepdims=True)        #normalize transition by row

    else:
        mus = []
        sigmas = []
        transitions = []
        initials = []
        with open(args.clusters_file,'r') as f:
            for line in f:
                #each line is a cluster, and looks like this:
                #initial mu_1 mu_2 sigma_0_0 sigma_0_1 sigma_1_0 sigma_1_1 transition_this_to_0 transition_this_to_1 ... transition_this_to_K-1
                vals = list(map(float,line.split()))
                initials.append(vals[0])
                mus.append(vals[1:3])
                sigmas.append([vals[3:5],vals[5:7]])
                transitions.append(vals[7:])
        initials = np.asarray(initials)
        transitions = np.asarray(transitions)
        mus = np.asarray(mus)
        sigmas = np.asarray(sigmas)
        args.cluster_num = len(initials)

    #TODO: Do whatever you want to pack mus, sigmas, initals, and transitions into the model variable (just a tuple, or a class, etc.)
    model = (initials, transitions, mus, sigmas)
    return model


def forward(model, data, args):
    from scipy.stats import multivariate_normal
    from math import log
    alphas = np.zeros((len(data),args.cluster_num))
    log_likelihood = 0.0
    #TODO: Calculate and return forward probabilities (normalized at each timestep; see next line) and log_likelihood
    #NOTE: To avoid numerical problems, calculate the sum of alpha[t] at each step, normalize alpha[t] by that value, and increment log_likelihood by the log of the value you normalized by. This will prevent the probabilities from going to 0, and the scaling will be cancelled out in train_model when you normalize (you don't need to do anything different than what's in the notes). This was discussed in class on April 3rd.
    initials, transitions, mus, sigmas = extract_parameters(model)
    N = len(data)
    K = args.cluster_num
    for n in range(N):
        if n == 0:
            for i in range(K): #alpha=initial*emission
                if args.tied:
                    alphas[n, i] = initials[i] * multivariate_normal.pdf(x=data[n], mean=mus[i], cov=sigmas)
                else:
                    alphas[n, i] = initials[i] * multivariate_normal.pdf(x=data[n], mean=mus[i], cov=sigmas[i])
        else:
            for i in range(K): #alpha=alpha(n-1)*transition*emission
                if args.tied:
                    alphas[n, i] = np.sum(alphas[n - 1, :] * transitions[:, i]) * multivariate_normal.pdf(x=data[n], mean=mus[i], cov=sigmas)
                else:
                    alphas[n, i] = np.sum(alphas[n - 1, :] * transitions[:, i]) * multivariate_normal.pdf(x=data[n], mean=mus[i], cov=sigmas[i])

        Z = np.sum(alphas[n, :])
        alphas[n, :] = alphas[n, :]/np.sum(alphas[n, :]) #normalize alphas
        log_likelihood = log_likelihood + log(Z)
    return alphas, log_likelihood

def backward(model, data, args):
    from scipy.stats import multivariate_normal
    betas = np.zeros((len(data),args.cluster_num))
    #TODO: Calculate and return backward probabilities (normalized like in forward before)
    initials, transitions, mus, sigmas = extract_parameters(model)
    K = args.cluster_num
    N = len(data)-1
    betas[N, :] = 1 #beta(n)=1
    for n in reversed(range(N)):
        for i in range(K):
            beta = 0
            for j in range(K): #beta(n+1)*transition*emission
                if args.tied:
                    beta += betas[n + 1, j] * transitions[i, j] * multivariate_normal.pdf(x=data[n + 1], mean=mus[j], cov=sigmas)
                else:
                    beta += betas[n + 1, j] * transitions[i, j] * multivariate_normal.pdf(x=data[n + 1], mean=mus[j], cov=sigmas[j])
            betas[n, i] = beta
        betas[n, :] = betas[n, :] / np.sum(betas[n, :]) #normalize betas
    return betas

def train_model(model, train_xs, dev_xs, args):
    from scipy.stats import multivariate_normal
    #TODO: train the model, respecting args (note that dev_xs is None if args.nodev is True)
    initials, transitions, mus, sigmas = extract_parameters(model)
    K = args.cluster_num
    N = train_xs.shape[0]
    for iteration in range(args.iterations):
        ksi = np.zeros((N, K, K))
        alphas, _ = forward(model, train_xs, args)
        betas = backward(model, train_xs, args)
        gammas = np.zeros((N, K))
        #E-step
        for n in range(N):
            for i in range(K):
                for j in range(K):
                    if n != 0:
                        if args.tied:
                            ksi[n, i, j] = alphas[n - 1, i] * betas[n, j] * transitions[i, j] * multivariate_normal.pdf(x=train_xs[n], mean=mus[j], cov=sigmas)
                        else:
                            ksi[n, i, j] = alphas[n - 1, i] * betas[n, j] * transitions[i, j] * multivariate_normal.pdf(x=train_xs[n], mean=mus[j], cov=sigmas[j])
                gammas[n, i] = alphas[n, i] * betas[n, i]
            gammas[n, ] = gammas[n, ]/np.sum(gammas[n, ]) #normalize gammas
            if n != 0:
                ksi[n, ] = ksi[n, ]/np.sum(ksi[n, ]) #normalize ksi
        #M-step
        for i in range(K):
            initials[i] = gammas[0, i]
            Z = np.sum(gammas[:, i])
            mus[i] = np.dot(gammas[:, i], train_xs) / Z
            if args.tied:
                sigmas += np.dot(gammas[:, i] * (train_xs - mus[i]).T, (train_xs - mus[i]))
            else:
                sigmas[i] = np.dot(gammas[:, i] * (train_xs - mus[i]).T, (train_xs - mus[i])) / Z
            for j in range(K):
                transitions[i,j] = np.sum(ksi[:,i,j])/np.sum(ksi[:,i,:])

        if args.tied:
            sigmas = sigmas/N
    model = (initials, transitions, mus, sigmas)

    return model


def average_log_likelihood(model, data, args):
    #TODO: implement average LL calculation (log likelihood of the data, divided by the length of the data)
    #NOTE: yes, this is very simple, because you did most of the work in the forward function above
    ll = 0.0
    _, log_likelihood = forward(model, data, args)
    ll = log_likelihood / len(data)
    return ll

def extract_parameters(model):
    #TODO: Extract initials, transitions, mus, and sigmas from the model and return them (same type and shape as in init_model)
    initials = model[0]
    transitions = model[1]
    mus = model[2]
    sigmas = model[3]

    return initials, transitions, mus, sigmas

def main():
    import argparse
    import os
    print('Gaussian') #Do not change, and do not print anything before this.
    parser = argparse.ArgumentParser(description='Use EM to fit a set of points')
    init_group = parser.add_mutually_exclusive_group(required=True)
    init_group.add_argument('--cluster_num', type=int, help='Randomly initialize this many clusters.')
    init_group.add_argument('--clusters_file', type=str, help='Initialize clusters from this file.')
    parser.add_argument('--nodev', action='store_true', help='If provided, no dev data will be used.')
    parser.add_argument('--data_file', type=str, default=os.path.join(DATA_PATH, 'points.dat'), help='Data file.')
    parser.add_argument('--print_params', action='store_true', help='If provided, learned parameters will also be printed.')
    parser.add_argument('--iterations', type=int, default=1, help='Number of EM iterations to perform')
    parser.add_argument('--tied',action='store_true',help='If provided, use a single covariance matrix for all clusters.')
    args = parser.parse_args()
    if args.tied and args.clusters_file:
        print('You don\'t have to (and should not) implement tied covariances when initializing from a file. Don\'t provide --tied and --clusters_file together.')
        exit(1)

    train_xs, dev_xs = parse_data(args)
    model = init_model(args)
    model = train_model(model, train_xs, dev_xs, args)
    nll_train = average_log_likelihood(model, train_xs, args)
    print('Train LL: {}'.format(nll_train))
    if not args.nodev:
        nll_dev = average_log_likelihood(model, dev_xs, args)
        print('Dev LL: {}'.format(nll_dev))
    initials, transitions, mus, sigmas = extract_parameters(model)
    if args.print_params:
        def intersperse(s):
            return lambda a: s.join(map(str,a))
        print('Initials: {}'.format(intersperse(' | ')(np.nditer(initials))))
        print('Transitions: {}'.format(intersperse(' | ')(map(intersperse(' '),transitions))))
        print('Mus: {}'.format(intersperse(' | ')(map(intersperse(' '),mus))))
        if args.tied:
            print('Sigma: {}'.format(intersperse(' ')(np.nditer(sigmas))))
        else:
            print('Sigmas: {}'.format(intersperse(' | ')(map(intersperse(' '),map(lambda s: np.nditer(s),sigmas)))))

if __name__ == '__main__':
    main()