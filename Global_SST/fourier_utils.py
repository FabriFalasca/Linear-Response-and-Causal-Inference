# Function for the computation of the Fourier Power Spectrum of a time series
# and testing its significance.
# It has been shown that the null hypothesis of sea surface temperature variability
# is red noise. Therefore, to test the significance of peaks in the power spectrum of a time series
# we (a) generate a large number of AR(1) processes, (b) compute the power spectrum of each one
# and (c) retain, for each frequency, only the lower and upper 5% percentiles.

# Fabri Falasca
# fabrifalasca@gmail.com

import numpy as np
import numpy.ma as ma
import numpy as np
from scipy.fft import fft
import random

def fourier_spectra(timeseries):
    # length of the time series
    length = len(timeseries)
    # array of times
    time_array = np.arange(1,length+1)
    # time increments
    tinc = time_array[1] - time_array[0]
    # frequency increments
    finc = 1/(tinc*length)
    # Fourier Transform
    ft = 1/length * np.abs(fft(timeseries))**2
    ff = np.arange(0,1/tinc,finc)
    # Power Spectrum
    ps = np.transpose([ff,ft])
    # (a) Remove the first point which is period = infinity
    # (b) Consider only up to half the spectrum as it is symmetric
    ps = ps[1:length//2]
    return ps

# From months to years
# Input: spectra computed from ts
def month_to_year(spectrum): return np.transpose([1/(spectrum[:,0]*12),spectrum[:,1]])

def lagged_correlation(x,y,tau):

    assert len(x) == len(y);
    #length of time series
    T = len(x);

    ##assert that lag can nit be greater than T
    assert tau < T;

    ##reduce time series to zero mean and unit variance
    x = (x-np.mean(x))/np.std(x);
    y = (y-np.mean(y))/np.std(y);


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

def lagged_autocorrelation(ts,tau):
    return lagged_correlation(ts,ts,tau);

# Generate an autoregressive process of order 1 from a time series
def ar1(timeseries):
    # length of the time series
    length = len(timeseries)
    # mean
    mean = np.mean(timeseries)
    # variance
    var = np.var(timeseries)
    # Autocorrelation of x(t) at lag tau = 1
    phi = lagged_autocorrelation(timeseries,1)

    #
    # AR(1) process: x(t) = delta + phi x(t-1) + w(t)
    #
    # Find the parameters

    # delta
    delta = mean * (1 - phi)
    # variance of the white noise w(t)
    var_white_noise = var * (1 - phi**2)
    # std of the white noise w(t)
    std_white_noise = np.sqrt(var_white_noise)

    # Generate the AR(1) process
    x = np.array([])
    # Initial condition is a random point taken from the timeseries
    x_in = np.array([timeseries[random.randint(0, length-1)]])
    for i in range(length):
        x_plus = delta + phi * x_in + np.random.normal(mean, std_white_noise, 1)
        x = np.append(x,x_plus)
        x_in = x_plus

    return x

# main function to test the significance of a given spectra
# inputs:
# (a) time series
# (b) sample: number of processes to compute
def spectral_significance(timeseries,sample):

    # Generate a (large) sample of AR(1) processes
    ar_1_processes = []
    for i in range(sample):
        ar_1_processes.append(ar1(timeseries))
    ar_1_processes = np.array(ar_1_processes)

    # For each process compute the power spectra
    spectra_ar1 = []
    for i in range(len(ar_1_processes)):
        spectra_ar1.append(fourier_spectra(ar_1_processes[i]))
    spectra_ar1 = np.array(spectra_ar1)

    # Consider all values of power density for a given frequency
    value_for_frequency = np.transpose(spectra_ar1[:,:,1])

    # For each one of these values find the lower and upper 5% percentile
    lower_percentiles = []
    higher_percentiles = []
    for i in range(len(value_for_frequency)):
        lower_percentiles.append(np.percentile(value_for_frequency[i], 5))
        higher_percentiles.append(np.percentile(value_for_frequency[i], 95))
    lower_percentiles = np.transpose([spectra_ar1[0,:,0],np.array(lower_percentiles)])
    higher_percentiles = np.transpose([spectra_ar1[0,:,0],np.array(higher_percentiles)])

    return lower_percentiles, higher_percentiles
