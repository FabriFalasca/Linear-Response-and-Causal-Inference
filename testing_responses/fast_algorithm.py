import numpy as np
from numpy.linalg import inv
from time import time
import scipy.stats

def norm(x,y):
    return np.abs(x-y).mean() / np.abs(y).mean()

def scale(x):
    x = x - np.mean(x,-1,keepdims=True)
    return x / np.std(x,-1,keepdims=True)

def AR1(x_t,phi,sigmas):
    '''
    Generate autoregression process
    x(N,T)
    initialized from random point in x_t
    where
    N - number of variables
    T - number of time points
    phi(N) - lag-1 correlation
    sigmas(N) - std in equilibrium
    '''
    N, T = x_t.shape
    
    x = np.zeros_like(x_t)
    #x[:,-1] to reproduce Fabri code; but should be x[:,0]
    x[:,-1] = x_t[:, np.random.randint(0, T-1)] # T-1?

    epsilon = sigmas*np.sqrt(1-(phi**2))
    
    for t in range(0,T): #range(0,T) to reproduce Fabri code; but should be range(1,T)
        x[:,t] = phi * x[:,t-1] + np.random.normal(0, epsilon)
    
    return x

def compute_quantile(x_t,phi,sigmas,tau,Ens):
    '''
    Input:
        x_t - original time series
        phi - lag-1 autocorrelation of time series
        sigmas - standard deviations of time series
        tau - number of lags to compute
        Ens - Ensemble size
    Returns:
        0.01 and 0.99 quantiles of size (tau,N,N)
    '''
    N, T = x_t.shape

    el_memory = Ens*N*T+2*Ens*N**2+2*tau*N**2
    print('Expected memory consumption in GB:', el_memory * 8 / 1024**3)
    
    print('Memory is allocating...')
    X   = 1.*np.zeros((Ens,N,T)) # trajectory
    iC  = 1.*np.zeros((Ens,N,N)) # inverse matrices
    q01 = 1.*np.zeros((tau,N,N)) # quantiles
    q99 = 1.*np.zeros((tau,N,N))
    R   = 1.*np.zeros((Ens,N,N)) # Response
    
    print('Memory is allocated')
    
    for ens in range(Ens):
        tt = time()
        X[ens] = scale(AR1(x_t,phi,sigmas))
        iC[ens] = inv(X[ens]@X[ens].T / T)
        print(f'Trajectory and inverse {ens+1}/{Ens} are computed for {round(time()-tt,2)} sec', end="\r")

    print('\nTrajectories and inverse matrices are ready')
    
    for t in range(tau):
        tt = time()
        for ens in range(Ens):
            C = X[ens,:,t:]@X[ens,:,:T-t].T / T
            R[ens] = C @ iC[ens]
        if t == 0:
            E = np.eye(N)
            if norm(R[ens],E) > 1e-3:
                print('Error: matrix is not invertible!')
        q01[t], q99[t] = list(np.quantile(R,[0.01,0.99],axis=0)) 
        print(f'Lag {t+1}/{tau} is computed for {round(time()-tt,2)} sec', end="\r")
    
    print('\nQuantiles are computed')
    return q01, q99

def compute_quantile_analytical(x_t,phi,sigmas,tau,Ens):
    N, T = x_t.shape
    q01 = 1.*np.zeros((tau,N,N)) # quantiles
    q99 = 1.*np.zeros((tau,N,N))

    tau_hat = np.nan_to_num(-2 / np.log(phi[:,None] * phi[None,:]), nan=0)
    tau_hat = np.maximum(tau_hat,1)

    for t in range(tau):
        q01[t] = np.diag(phi**t)
        q99[t] = np.diag(phi**t)

    q01 -= 2.32 * np.sqrt(tau_hat/T)
    q99 += 2.32 * np.sqrt(tau_hat/T)
            
    return q01, q99

def compute_quantile_analytical_tau(x_t,phi,sigmas,tau,Ens,standardized='yes',q=[0.01,0.99]):
    N, T = x_t.shape
    q0 = 1.*np.zeros((tau,N,N)) # quantiles
    var = 1.*np.zeros((tau,N,N))

    variance = sigmas**2

    if standardized == 'yes':
        variance = np.ones_like(variance)

    phi_i = np.repeat(phi[:,None],N,axis=1)
    phi_j = np.repeat(phi[None,:],N,axis=0)
    v_i   = np.repeat(variance[:,None],N,axis=1)
    v_j   = np.repeat(variance[None,:],N,axis=0)
    
    Phi = phi_i * phi_j
    mask = phi_i!=phi_j # Mask of points where log(phi_i/phi_j) does not lead to division by zero
    maskn = np.logical_not(mask)

    # Limit the minimum correlation 
    Phi_truncated = np.maximum(Phi, np.exp(-2))

    for t in range(tau):
        q0[t] = np.diag(phi**t)

        var[t] = - 2 / T * (1 - Phi**t) / np.log(Phi_truncated)
        var[t,mask] -= 2/T * (phi_i[mask]**t * (phi_i[mask]**t-phi_j[mask]**t)/np.log(phi_i[mask]/phi_j[mask]))
        var[t,maskn] -= 2/T * phi_i[maskn] *  t * phi_i[maskn]**t
        
    var *= v_i / v_j # Simple formula to account for different variances

    for _q in q:
        yield q0 + scipy.stats.norm.ppf(_q) * np.sqrt(var)

def compute_quantile_analytical_tau_discrete(x_t,phi,sigmas,tau,Ens,standardized='yes', q=[0.01,0.99]):
    N, T = x_t.shape

    print('Memory demands: ', 2*N**2*tau*4/1024**3, 'GB')
    q01 = np.zeros((tau,N,N), dtype='float32') # The first quantile
    q99 = np.zeros((tau,N,N), dtype='float32') # The second quantile

    variance = sigmas**2

    if standardized == 'yes':
        variance = np.ones_like(variance)

    phi_i = np.repeat(phi[:,None],N,axis=1)
    phi_j = np.repeat(phi[None,:],N,axis=0)
    v_i   = np.repeat(variance[:,None],N,axis=1)
    v_j   = np.repeat(variance[None,:],N,axis=0)
    vi_vj = v_i / v_j
    
    mask = phi_i!=phi_j
    maskn = np.logical_not(mask)

    # Limit the minimum correlation
    Phi = phi_i * phi_j
    factor_1 = (1+Phi)/(1-Phi) * 1/T
    factor_2 = 1 /(1-Phi) * 2/T
    with np.errstate(divide='ignore'):
        factor_3 = phi_i / (phi_j-phi_i) * 2/T
        factor_3[maskn] = 0.

    phi_t = np.ones(phi.shape)
    tt = time()
    for t in range(tau):
        # Scalar calculations
        phi_i_t = np.repeat(phi_t[:,None],N,axis=1)
        phi_j_t = np.repeat(phi_t[None,:],N,axis=0)
        phi_i_2t = np.repeat((phi_t**2)[:,None],N,axis=1)

        q0 = np.diag(phi_t)

        var = (phi_i_2t+1) * factor_1
        var -= phi_i_t * (phi_j_t+phi_i_t*Phi) * factor_2
        var -= phi_i_t * (phi_j_t-phi_i_t) * factor_3
        var[maskn] -= (2/T * t) * phi_i_2t[maskn]
        var *= vi_vj
        var = np.maximum(var, 0)

        phi_t *= phi

        q01[t] = q0 + scipy.stats.norm.ppf(q[0]) * np.sqrt(var)
        q99[t] = q0 + scipy.stats.norm.ppf(q[1]) * np.sqrt(var)
        time_so_far = time() - tt
        time_expected = time_so_far * tau / (t+1)
        time_per_iteration = time_so_far / (t+1)
        print(f'Iteration [{t+1}/{tau}], one iteration is {round(time_per_iteration,2)} seconds, time progress [{round(time_so_far,2)}/{round(time_expected,2)}]                  ', end='\r')

    print('',end='\n')
        
    return q01, q99

def compute_quantile_analytical_tau_discrete_old(x_t,phi,sigmas,tau,Ens,standardized='yes', q=[0.01,0.99]):
    N, T = x_t.shape
    q0 = 1.*np.zeros((tau,N,N)) # quantiles
    var = 1.*np.zeros((tau,N,N))

    variance = sigmas**2

    if standardized == 'yes':
        variance = np.ones_like(variance)

    phi_i = np.repeat(phi[:,None],N,axis=1)
    phi_j = np.repeat(phi[None,:],N,axis=0)
    v_i   = np.repeat(variance[:,None],N,axis=1)
    v_j   = np.repeat(variance[None,:],N,axis=0)
    
    Phi = phi_i * phi_j
    mask = phi_i!=phi_j # Mask of points where log(phi_i/phi_j) does not lead to division by zero
    maskn = np.logical_not(mask)

    tt = time()
    for t in range(tau):
        q0[t] = np.diag(phi**t)

        var[t] = 1/T * (1+phi_i**(2*t))*(1+Phi)/(1-Phi)
        var[t] -= 2/T * phi_i**t * (phi_j**t+phi_i**t*Phi)/(1-Phi)

        var[t,mask] -= 2/T * phi_i[mask]**t * phi_i[mask] * (phi_j[mask]**t-phi_i[mask]**t)/(phi_j[mask]-phi_i[mask])
        var[t,maskn] -= 2/T * phi_i[maskn]**t * t * phi_i[maskn]**t
        time_so_far = time() - tt
        time_expected = time_so_far * tau / (t+1)
        time_per_iteration = time_so_far / (t+1)
        print(f'Iteration [{t+1}/{tau}], one iteration is {round(time_per_iteration,2)} seconds, time progress [{round(time_so_far,2)}/{round(time_expected,2)}]                  ', end='\r')

    var *= v_i / v_j # Simple formula to account for different variances

    print('',end='\n')
        
    for _q in q:
        yield q0 + scipy.stats.norm.ppf(_q) * np.sqrt(var)

if __name__ == "__main__":
    import utils_linear_response
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--T', type=int, default=7800)
    parser.add_argument('--N', type=int, default=1000)
    parser.add_argument('--Ens', type=int, default=1000)
    parser.add_argument('--tau', type=int, default=3)
    args = parser.parse_args()
    print(args)

    def truncate_range(x, eps=0.01):
        return np.minimum(np.maximum(x, eps), 1-eps)

    def compute_quantile_original(x_t,phi,sigmas,tau,Ens,standardized='yes'):
        null_responses_samples = utils_linear_response.null_responses(x_t,phi,tau,sigmas,Ens,standardized)
        q01 = np.quantile(null_responses_samples,0.01,axis = 0)
        q99 = np.quantile(null_responses_samples,0.99,axis = 0)
        return q01, q99

    def driver(T=args.T, N=args.N, Ens=args.Ens, tau=args.tau, fun=compute_quantile_original):
        np.random.seed(0)
        phi = truncate_range(np.random.rand(N))
        sigmas = truncate_range(np.random.rand(N))
        x_t = np.random.randn(N,T)
        print('Inputs are ready')
        return fun(x_t,phi,sigmas,tau,Ens)

    print('\n\n ------------------------')
    print('\nAnalytical estimate discrete new:')
    tt = time()
    x, y = driver(fun=compute_quantile_analytical_tau_discrete)
    print('Time = ', time() - tt)

    print('\nAnalytical estimate discrete old:')
    tt = time()
    x1, y1 = driver(fun=compute_quantile_analytical_tau_discrete_old)
    print('Time = ', time() - tt)
    
    print('Error = ', norm(x,x1), norm(y,y1))

    print('\n\n ---------Not standardized----------')
    print('\nAnalytical estimate discrete new:')
    x, y = driver(fun=lambda *x: compute_quantile_analytical_tau_discrete(*x, standardized='no'))

    print('\nAnalytical estimate discrete old:')
    x1, y1 = driver(fun=lambda *x: compute_quantile_analytical_tau_discrete_old(*x, standardized='no'))
    print('Error = ', norm(x,x1), norm(y,y1))