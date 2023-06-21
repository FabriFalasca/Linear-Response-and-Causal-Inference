import numpy as np
from numpy.linalg import inv
from time import time

def norm(x,y):
    if isinstance(x, np.ndarray):
        return np.abs(x-y).mean() / np.abs(y).mean()
    elif isinstance(x, torch.Tensor):
        return torch.abs(x-y).mean() / torch.abs(y).mean()
    else:
        print('I am sorry')

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

import torch

def compute_quantile(x_t,phi,sigmas,tau,Ens,dtype=torch.float32,device=None):
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
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('The following device will be used:', device)

    N, T = x_t.shape
    byte = 8 if dtype==torch.float64 else 4

    memory = byte * (Ens*N*T + 2*Ens*N**2) + 4 * 2*tau*N**2
    print('Expected memory consumption in GB:', memory / 1024**3)
    
    print('Memory is allocating...')
    X  = torch.zeros(Ens,N,T,dtype=dtype)
    iC = torch.zeros(Ens,N,N,dtype=dtype)
    R  = torch.zeros(Ens,N,N,dtype=dtype,device=device)

    q01 = 1.*np.zeros((tau,N,N), dtype='float32')
    q99 = 1.*np.zeros((tau,N,N), dtype='float32')
    
    print('Memory is allocated')
    
    for ens in range(Ens):
        tt = time()
        X[ens] = torch.tensor(scale(AR1(x_t,phi,sigmas)),dtype=dtype)
        Mat = X[ens].to(device)
        iC[ens] = torch.inverse(Mat@Mat.T).cpu()
        print(f'Trajectory and inverse {ens+1}/{Ens} are computed for {round(time()-tt,2)} sec', end="\r")

    print('\nTrajectories and inverse matrices are ready')
    
    for t in range(tau):
        tt = time()
        for ens in range(Ens):
            Mat1 = X[ens].to(device)
            Mat2 = iC[ens].to(device)
            R[ens] = (Mat1[:,t:]@Mat1[:,:T-t].T) @ Mat2
            if t == 0:
                E = torch.eye(N,dtype=dtype,device=device)
                if norm(R[ens],E).cpu() > 1e-3:
                    print('Error: matrix is not invertible!')
        q01[t], q99[t] = list(torch.quantile(R,torch.tensor([0.01,0.99],dtype=dtype,device=device),dim=0).to('cpu').numpy()) 
        print(f'Lag {t+1}/{tau} is computed for {round(time()-tt,2)} sec', end="\r")
    
    print('\nQuantiles are computed')
    return q01, q99

if __name__ == "__main__":
    import utils_linear_response
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--T', type=int, default=8000)
    parser.add_argument('--N', type=int, default=3400)
    parser.add_argument('--Ens', type=int, default=1)
    parser.add_argument('--tau', type=int, default=2)
    args = parser.parse_args()
    print(args)

    def truncate_range(x, eps=0.1):
        return np.minimum(np.maximum(x, eps), 1-eps)

    def timer(func):
        # This function shows the execution time of 
        # the function object passed
        def wrap_func(*args, **kwargs):
            t1 = time()
            result = func(*args, **kwargs)
            t2 = time()
            print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
            return result
        return wrap_func

    def compute_quantile_original(x_t,phi,sigmas,tau,Ens):
        null_responses_samples = utils_linear_response.null_responses(x_t,phi,tau,sigmas,Ens,'yes')
        q01 = np.quantile(null_responses_samples,0.01,axis = 0)
        q99 = np.quantile(null_responses_samples,0.99,axis = 0)
        return q01, q99
    
    @timer
    def driver(T=args.T, N=args.N, Ens=args.Ens, tau=args.tau, fun=compute_quantile_original):
        np.random.seed(0)
        phi = truncate_range(np.random.rand(N))
        sigmas = truncate_range(np.random.rand(N))
        x_t = np.random.randn(N,T)
        return fun(x_t,phi,sigmas,tau,Ens)
    
    for i in range(2):
        print('\n\n ------------------------')
        print('\nOriginal code:')
        x0, y0 = driver()

        print('\nTorch code:')
        x, y = driver(fun=compute_quantile)
        print('Error: ', norm(x,x0), norm(y,y0))