import sys
import os
import time
import numpy as np
import scipy.stats
import sklearn

from numpy.linalg import inv

'''
Author: Fabri Falasca; fabrifalasca@gmail.com

In this script you find some useful functions to explore and test causality from time series data
in the framework of linear response theory.

Functions:

--- REMOVE MEAN

# Function: remove_mean(x_t)
Given a set of n time series of length T ( np.shape(x_t) = (n,T) ) it scales each xt
to zero mean.

--- STANDARDIZE

# Function: remove_std(x_t)
Given a set of n time series of length T ( np.shape(x_t) = (n,T) ) it it scales each xt
to unit variance.

--- COMPUTE LAG CORRELATIONS

# Function: lagged_correlation(x,y,tau)
Given two time series x and y, it computes the correlation at lag tau.
Note: time series are supposed to be standardized to zero mean and unit variance.

--- LAG 1 AUTOCORRELATIONS

# Function: phi_vector(x_t)
Given a multidimensional time series x_t, it computes the lag-1 autocorrelation of
each time series. x_t should have this "shape": np.shape(x_t) = (dimension, length)
Where "dimension" is the number of time series and "length" is the total number of time steps.

--- COMPUTE STANDARD DEVIATIONS

# Function: sigmas(x_t)
Given a multidimensional time series x_t, it computes the standard deviation of
each time series. x_t should have this "shape": np.shape(orbit) = (dimension, length)
Where "dimension" is the number of time series and "length" is the total number of time steps.

--- NULL MODEL FOR STATISTICAL SIGNIFICANCE

# Function: linear_markov_null_model(x_t,phi,sigmas)
Definition of the null model. Given a time x_t with lag-1 autocorrelation phi and standard deviations sigmas, this function defines
the null model useful for computing statistical significance. The format of time series is np.shape(x_t) = (dimension, length) 
where "dimension" is the number of time series and "length" is the total number of time steps.

--- COMPUTE RESPONSES GIVE AN ORBIT x_t

# Function: response(x_t,tau_max,standardized)
Given a multidimensional time series x_t it computes the response matrix up to a lag tau_max.
If standardized == 'yes' it estimates correlations.  If standardized == 'no' it estimates covariances. 

--- GENERATE AN ENSEMBLE OF NULL MODELS

# Function: null_responses(x_t,phi,tau_max,R,standardized)
Function to generate R realizations of the proposed null model.
'''

########## REMOVE MEAN

# x_t is an array with n time series of length T
# format: np.shape(xt) = (n, T)

def remove_mean(x_t):

    # x_t is an array with n time series of length T
    # format: np.shape(xt) = (n, T)
    d = np.shape(x_t)[0]
    y_t = x_t.copy()
    for i in range(d):
        y_t[i] = y_t[i] - np.mean(y_t[i])
    return y_t

########## STANDARDIZE

# x_t is an array with n time series of length T
# format: np.shape(xt) = (n, T)

def standardize(x_t):

    # xt is an array with n time series of length T
    # format: np.shape(xt) = (n, T)
    d = np.shape(x_t)[0]
    y_t = x_t.copy()
    for i in range(d):
        y_t[i] = y_t[i]/np.std(y_t[i])
    return y_t

########## COMPUTE LAG CORRELATIONS

# Here we are assuming that time series have been rescaled to
# zero mean and unit variance

def lagged_correlation(x,y,tau):

    # Check that the two time series have
    # - the same length
    assert len(x) == len(y);
    #length of time series
    T = len(x);

    ##assert that lag can nit be greater than T
    assert tau < T;

    if(tau == 0):
        return np.dot(x,y)/T;
    if(tau > 0):
        x = x[0:T-tau];
        y = y[tau:T]
        return np.dot(x,y)/T;

    if(tau < 0):
        tau = np.abs(tau)
        y = y[0:T-tau];
        x = x[tau:T]
        return np.dot(x,y)/T;

########## LAG 1 AUTOCORRELATIONS

# Compute lag-1 autocorrelation of an orbit x_t
# Orbit must have this shape: np.shape(x_t) = (dimension, length)

def phi_vector(x_t):
    # we scale the process to zero mean
    x_t = remove_mean(x_t)
    # we scale the process to unit variance
    x_t = standardize(x_t)
    # Compute lag-1 correlation
    phi = []
    for i in range(len(x_t)):
        phi.append(lagged_correlation(x_t[i],x_t[i],1))
    phi = np.array(phi)
    return phi

########## COMPUTE STANDARD DEVIATIONS

# Compute the sigmas of each time series in xt
# Orbit must have this shape: np.shape(x_t) = (dimension, length)

def sigmas(x_t):
    # we scale the process to zero mean
    x_t = remove_mean(x_t)
    # Compute standard deviation of each time series
    sigmas = np.std(x_t,axis = 1)
    return sigmas

########## NULL MODEL FOR STATISTICAL SIGNIFICANCE

# For autoregressive processes

import random

def linear_markov_null_model(x_t,phi,sigmas):

    # Dimensionality of the process: how many time series?
    d = np.shape(x_t)[0]
    # Period: length of the time series?
    period = np.shape(x_t)[1]

    ########## Transition matrix
    # It is a diagnal matrix defined by lag-1 autocorrelations
    A_tilde = np.diag(phi, k=0)

    # Initialize null model
    x_ar1 = []

    # Initial point
    x_in = x_t[:,np.random.randint(0, period-1)]

    # sigma of the noise
    sigma_epsilon = sigmas*np.sqrt(1-(phi**2))

    # forward integration
    for t in range(period):
        x_plus = np.matmul(A_tilde,x_in) + np.random.normal(0, sigma_epsilon, d)
        x_ar1.append(x_plus)
        x_in = x_plus

    x_ar1 = np.transpose(np.array(x_ar1))

    return x_ar1

########## COMPUTE RESPONSES GIVEN AN ORBIT x_t

### Compute response from an orbit x_t

# Inputs:

# - x_t: multi dimensional time series. Shape: np.shape(x_t) = (dimension, length)
# "dimension" is the number of time series and
# "length# is the length of each time series

# - tau_max: maximum lag to compute the response

# - standardized: 'yes' or 'no'
# if "yes" than the time series are standardized to unit variance. Responses are then computed using
# correlation functions
# if "no", responses are then computed using the covariance functions. 

def response(x_t,tau_max,standardized):

    # we scale the process to zero mean
    x_t = remove_mean(x_t)

    # Time length
    n_time = np.shape(x_t)[1]
    # Number of time series
    n_ts = np.shape(x_t)[0]

    response_matrix = []

    if standardized == 'yes':
        # Standardize time series to unit variance
        x_t = standardize(x_t)

    # By doing so we compute correlation matrices

    ########## Compute covariance matrix at lag zero
    C_0 = np.dot(x_t,np.transpose(x_t))/(n_time)

    ########## Compute inverse of the covariance matrix
    inverse_C0 = inv(C_0)

    response_matrix = []

    for tau in range(tau_max):
        orbit1 = x_t[:,tau:]  # remove the FIRST tau elements
        orbit2 = x_t[:,:n_time-tau]  # remove the LAST tau elements
        # Compute the covariance matrix at lag tau
        C_tau = np.dot(orbit1,np.transpose(orbit2))/(n_time)
        ########## Compute the response matrix
        response_matrix.append(np.matmul(C_tau,inverse_C0))

    response_matrix = np.array(response_matrix)

    return response_matrix

########## GENERATE AN ENSEMBLE OF NULL MODELS

# Generate R null models with same
# mean, variance and lag-1 autocorrelation of the original time series
# and compute the response matrix for each sample
# Save the "null responses"

def null_responses(x_t,phi,tau_max,sigmas,R,standardized):

    # x_t : orbit with np.shape(x_t) = (dimension, length)
    # phi: lag 1 autocorrelations
    # tau_max : Largest lag for response computation
    # R: Number of samples
    # sigmas: standard deviation of each time series
    # standardized: if yes then time series are standardized to unit variance
    # before computing the response

    null_model_sample = []
    null_responses = []
    counter = 0;

    for j in range(R):

        null_model_sample = linear_markov_null_model(x_t,phi,sigmas)
        null_responses.append(response(null_model_sample,tau_max,standardized))

        counter += 1;
        if(counter%(R/10) == 0):
            print('Progress: '+str(np.round(counter/R,2))+ ' of 1');

    null_responses = np.array(null_responses)

    return null_responses
