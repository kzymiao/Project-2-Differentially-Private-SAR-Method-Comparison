import numpy as np
import scipy
from scipy import stats
import scipy.sparse as sps
from scipy.sparse import coo_matrix
from random import randint, sample
import networkx as nx
import time
import warnings

def project_pd_cone(matrix, eps_H=1e-7):
        """Helper function to project a symmetric matrix onto the PD cone."""
        # Compute eigenvalues and eigenvectors for the symmetric matrix
        evals, evecs = np.linalg.eigh(matrix)
        # Truncate eigenvalues below the strictly positive threshold eps_H
        evals = np.maximum(evals, eps_H)
        # Reconstruct and return the positive definite matrix
        return evecs @ np.diag(evals) @ evecs.T

def f_lambda_ture_sig(
    hbeta,
    hrho,
    Y,
    X,
    W_dense,
    WY,
): 
    """
    Calculate the first-order derivatives (score/gradient) and estimate the variance sigma^2 
    for the standard SAR QMLE method.
    
    Parameters:
    :hbeta: The estimate of beta in the current iterative step.
    :hrho: The estimate of rho in the current iterative step.
    :Y: The observed response vector.
    :X: The covariate matrix.
    :W_dense: dense form of the spatial weight matrix W; W_dense and WY are precomputed outside this function because they do not change across gradient iterations.
    :WY: W @ Y

    Returns:
    :f: The combined gradient vector for beta and rho.
    :hsig: The estimated error variance (sigma^2).
    """

    N = len(Y)

    # S(rho) = I - rho W.
    # The inverse is still required for the trace term.
    hS = np.eye(N) - hrho * W_dense
    hS1 = np.linalg.inv(hS)

    # Instead of computing
    #
    #     (I - rho W) @ Y
    #
    # at every iteration, use
    #
    #     Y - rho * (W @ Y),
    #
    # where WY is precomputed once.
    V = (
        Y
        - hrho * WY
        - X.dot(hbeta).reshape(N, 1)
    )

    # Current variance estimate.
    hsig = (V.T.dot(V) / N)[0, 0]

    # trace(W @ hS1) can be computed without forming the full
    # matrix product:
    #
    # trace(W @ hS1)
    # = sum_ij W_ij * hS1_ji.
    trace_term = np.sum(
        W_dense * hS1.T
    )

    # Score with respect to rho.
    f_r = (
        -trace_term
        + (1.0 / hsig) * V.T.dot(WY)
    )

    # Score with respect to beta.
    f_b = (
        (1.0 / hsig)
        * X.T.dot(V)
    )

    f = np.vstack((f_b, f_r))

    return f, hsig


def qmle_dp_estimate(X,Y,W,epsi,delta):
    """
    [Algorithm 1: Differentially Private SAR Model (Estimation)]
    Obtain estimator by DP-Gradient Descent algorithm.
    """
    N=len(Y)
    p=len(X[1])
    
    # Input: Initial value \theta^0
    hrbeta=(0.01*np.ones(p+1)).reshape((p+1,1))
    # Convert the spatial weight matrix once.
    
    W_dense = (
        W.toarray()
        if sps.issparse(W)
        else np.asarray(W)
    )
    k=0
    
    # Initial variance computation \sigma(0)^2
    WY_initial = W_dense.dot(Y)

    V = (
        Y
        - hrbeta[p,] * WY_initial
        - X.dot(hrbeta[0:p]).reshape(N, 1)
    )

    hsig = (
        V.T.dot(V) / N
    )[0, 0]
    
    # Input: Feasibility parameters (truncation boundaries) c_\beta and c_\rho
    cb=np.sqrt(p)
    cr=0.5
    
    # Input: Truncation level R for dependent variable y 
    cy=hsig*np.sqrt(2*np.log(N))
    
    # Apply truncation: \Pi_R(y)
    Y=np.clip(Y, -cy, cy)
    WY = W_dense.dot(Y)
    cx=3*np.sqrt(p)
    cl2=0.5*np.log(N)/hsig
    
    # Input: Step size \eta^0
    # Note: eta contains 1/N, effectively dividing the gradient sum by N as required by Alg 1.
    eta=3/(4*N*cl2)
    
    # Input: Number of iterations T
    T=0.8*np.log(N)**2
    
    w1=0.8*np.log(N)
    
    # Input: Noise scales B_\rho and B_\beta (Sensitivities)
    sen_r=(2*(1+w1)*cr/(1-cr)**2+12*cy**2*w1/hsig+4*cy*cx*cb/hsig+4*w1*cy*cx*cb/hsig+4*cy**2/hsig)
    sen_b=4*cx*cy/hsig+8*cx*cy*w1/hsig+4*cx**2*cb/hsig
    
    # Calculate variances for Gaussian noise v_t^\beta and v_t^\rho
    var_b=2*sen_b**2*eta**2*np.log(2*T/delta)*T**2/(N**2*epsi**2)
    var_r=2*sen_r**2*eta**2*np.log(2*T/delta)*T**2/(N**2*epsi**2)
    
    f,hsig=f_lambda_ture_sig(hrbeta[0:p,],hrbeta[p,],Y,X,W_dense,WY,)
    

    while k<=T:

        lr=eta/(hsig)
        
        # Generate independent Gaussian noise v_t^\beta and v_t^\rho
        v=np.vstack((np.random.normal(0,var_b**0.5,p).reshape(p,1),np.random.normal(0,var_r**0.5,1).reshape(1,1)))
        
        hrbeta+=lr*f+v
        
        if(np.linalg.norm(hrbeta[0:p,],ord=2,axis=None)>cb):
            hrbeta[0:p,]=cb*hrbeta[0:p,]/np.linalg.norm(hrbeta[0:p,],ord=2,axis=None)
            
        if(abs(hrbeta[p,])>cr):
            hrbeta[p,]=cr
            
        k+=1
        
        f,hsig=f_lambda_ture_sig(hrbeta[0:p,],hrbeta[p,],Y,X,W_dense,WY,)
    
    return hrbeta,hsig,T,var_b,var_r,cl2
