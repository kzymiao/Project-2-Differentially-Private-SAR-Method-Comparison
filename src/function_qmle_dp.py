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
    WY,
    W_eigenvalues,
):
    """
    Calculate the QMLE score and current variance estimate.

    The trace term is evaluated from the eigenvalues of W,
    avoiding repeated inversion of I - rho W.
    """
    N = len(Y)

    rho = float(
        np.asarray(hrho).squeeze()
    )

    V = (
        Y
        - rho * WY
        - X.dot(hbeta).reshape(N, 1)
    )

    hsig = (
        V.T.dot(V) / N
    )[0, 0]

    trace_complex = np.sum(
        W_eigenvalues
        / (
            1.0
            - rho * W_eigenvalues
        )
    )

    # W is real, so complex eigenvalues occur in conjugate pairs.
    # The theoretical trace is real; any small imaginary component
    # is numerical roundoff.
    if abs(trace_complex.imag) > (
        1e-8
        * max(
            1.0,
            abs(trace_complex.real),
        )
    ):
        warnings.warn(
            "Non-negligible imaginary component "
            "in eigenvalue-based trace calculation."
        )

    trace_term = float(
        trace_complex.real
    )

    f_r = (
        -trace_term
        + (1.0 / hsig)
        * V.T.dot(WY)
    )

    f_b = (
        (1.0 / hsig)
        * X.T.dot(V)
    )

    f = np.vstack(
        (f_b, f_r)
    )

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

    W_eigenvalues = np.linalg.eigvals(
    W_dense
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
    
    f, hsig = f_lambda_ture_sig(
    hrbeta[0:p,],
    hrbeta[p,],
    Y,
    X,
    WY,
    W_eigenvalues,)
    

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

        f, hsig = f_lambda_ture_sig(
            hrbeta[0:p,],
            hrbeta[p,],
            Y,
            X,
            WY,
            W_eigenvalues,
        )
        
    
    return hrbeta,hsig,T,var_b,var_r,cl2
