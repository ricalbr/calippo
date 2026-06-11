#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 12:07:46 2025

@author: Francesco Ceccarelli, Andrea Crespi

Utilities and helpers for UPP calibration/operation.

This module provides utility functions for measuring and processing unitaries
on a Universal Photonic Processor (UPP), including Sinkhorn normalization,
fidelity metrics, random unitary generation and other.

Functions
---------
sinkhorn(U, tol=1e-3, max_iter=1000)
    Normalize matrices to be doubly stochastic using Sinkhorn-Knopp.
amplitude_fidelity(U_target, U_exp)
    Calculate fidelity based on the magnitude of matrix elements.
fidelity(U_target, U_exp)
    Calculate the standard fidelity for complex unitaries.
make_real_border(U)
    Make a unitary matrix "real-bordered" by zeroing angles on 1st row/col.
get_haar_unitary(size)
    Generate a complex random Haar-distributed unitary matrix.
"""

import numpy as np
from scipy.stats import unitary_group
from typing import Tuple


def sinkhorn(
    U: np.ndarray, 
    tol: float = 1e-3, 
    max_iter: int = 1000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize a (batch of) positive matrices using the Sinkhorn–Knopp algorithm.

    The algorithm iteratively rescales the rows and columns of each matrix
    to make it approximately doubly stochastic (rows and columns sum to 1).
    It returns both the normalized matrix and the diagonal correction
    factors such that:

        U = D_out @ U_norm @ D_in

    Parameters
    ----------
    U : np.ndarray, shape (N, N) or (M, N, N)
        Input matrix or batch of matrices with strictly positive entries.
    tol : float, optional
        Convergence tolerance. Default is 1e-3.
    max_iter : int, optional
        Maximum number of iterations allowed. Default is 1000.

    Returns
    -------
    U_norm : np.ndarray
        Normalized matrix (or batch), approximately doubly stochastic.
    D_in : np.ndarray
        Right diagonal scaling factors.
    D_out : np.ndarray
        Left diagonal scaling factors.
    """
    # Promote single matrix to batch dimension if 2D
    is_2d = U.ndim == 2
    if is_2d:
        U_work = U[np.newaxis, ...].astype(float, copy=True)
    else:
        U_work = U.astype(float, copy=True)
    
    M, N, _ = U_work.shape
    D_in = np.ones((M, N))
    D_out = np.ones((M, N))
    
    for i in range(max_iter):
        # --- Column normalization ---
        col_sums = np.sum(U_work, axis=1)
        T_in = 1.0 / col_sums
        U_work = U_work * T_in[:, np.newaxis, :]
        D_in *= T_in
        
        # --- Row normalization ---
        row_sums = np.sum(U_work, axis=2)
        T_out = 1.0 / row_sums
        U_work = T_out[:, :, np.newaxis] * U_work
        D_out *= T_out
        
        # Check convergence
        err_col = np.max(np.abs(1 - col_sums))
        err_row = np.max(np.abs(1 - row_sums))
        if max(err_col, err_row) < tol:
            break
    else:
        print(f"[WARNING] Sinkhorn did not converge within {max_iter} iterations.")

    if is_2d:
        return U_work[0], D_in[0], D_out[0]
    return U_work, D_in, D_out


def amplitude_fidelity(U_target: np.ndarray, U_exp: np.ndarray) -> np.ndarray:
    """
    Calculate the amplitude fidelity between two complex matrices.

    This metric evaluates the similarity between the magnitudes of the 
    matrix elements, ignoring phase differences. It is useful when only 
    power measurements are available. Supports single (N, N) or batch 
    (M, N, N) matrices.
    
    Parameters
    ----------
    U_target : np.ndarray, shape (N, N) or (M, N, N)
        The ideal or target unitary matrix (or batch of matrices).
    U_exp : np.ndarray, shape (N, N) or (M, N, N)
        The experimental or measured matrix (or batch of matrices).

    Returns
    -------
    np.ndarray
        Amplitude fidelity values between 0 and 1 (scalar or array of shape (M,)).
    """
    return fidelity(np.abs(U_target), np.abs(U_exp))


def fidelity(U_target: np.ndarray, U_exp: np.ndarray) -> np.ndarray:
    """
    Calculate the fidelity between two complex unitary matrices.

    The fidelity is defined as the normalized absolute trace of the 
    overlap between the target and experimental matrices:
        
        F = |Tr(U_target^H @ U_exp)| / N.
    
    This metric is sensitive to relative phases but invariant to a 
    global phase shift. Supports both single (N, N) and batch (M, N, N) 
    matrices.
    
    Parameters
    ----------
    U_target : np.ndarray, shape (N, N) or (M, N, N)
        The ideal or target unitary matrix (or batch of matrices).
    U_exp : np.ndarray, shape (N, N) or (M, N, N)
        The experimental or measured matrix (or batch of matrices).

    Returns
    -------
    np.ndarray
        Fidelity values between 0 and 1 (scalar or array of shape (M,)).
    """
    N = U_target.shape[-1]
    
    # Calcolo dell'overlap (batch o singolo)
    overlap = np.sum(U_target.conj() * U_exp, axis=(-2, -1))
    
    # Restituisce l'array delle fedeltà
    return np.abs(overlap) / N


def make_real_border(U: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Make a unitary matrix "real-bordered".

    Calculates the diagonal unitaries Dleft and Dright such that
    R = Dleft @ U @ Dright has entirely real elements on its first row 
    and first column. The elements of the first row and first column 
    of the input matrix U must be non-vanishing.

    Parameters
    ----------
    U : np.ndarray, shape (N, N)
        The input unitary matrix to be modified.

    Returns
    -------
    R : np.ndarray
        The real-bordered unitary matrix.
    Dleft : np.ndarray
        The left diagonal unitary correction matrix.
    Dright : np.ndarray
        The right diagonal unitary correction matrix.
    """
    Dright = np.diag(U[0, :].conj() / np.abs(U[0, :]))
    U1 = U @ Dright
    Dleft = np.diag(U1[:, 0].conj() / np.abs(U1[:, 0]))
    
    return Dleft @ U1, Dleft, Dright


def get_haar_unitary(size: int) -> np.ndarray:
    """
    Generate a random unitary matrix sampled from the Haar measure.

    Parameters
    ----------
    size : int
        The dimension N of the NxN unitary matrix.

    Returns
    -------
    np.ndarray
        A complex NxN Haar-random unitary matrix.
    """
    return unitary_group.rvs(size)

# ==================================================================
# Entry Point
# ==================================================================
if __name__ == "__main__":
    
    try:
        # Configuration
        size = 24
        noise_intensity = 0.1
        
        # 1. Create target random Haar matrix
        U_target = get_haar_unitary(size)
        
        # 2. Add noise (simulating experimental degradation)
        # Note: adding noise directly to complex values breaks unitarity
        noise = (np.random.randn(size, size) + 1j*np.random.randn(size, size)) * noise_intensity
        U_noisy = U_target + noise
        
        # 3. Use Sinkhorn to recover a doubly stochastic matrix
        # Note: we apply Sinkhorn to the power matrix (absolute squares) to preserve normalization
        U_power = np.abs(U_noisy)**2
        U_norm_power, _, _ = sinkhorn(U_power)
        
        # 4. Reconstruct normalized complex matrix (preserving noisy phases)
        U_restored = np.sqrt(U_norm_power) * np.exp(1j * np.angle(U_noisy))
        
        # 5. Calculate fidelity metrics
        fid = fidelity(U_target, U_restored)
        fid_amp = amplitude_fidelity(U_target, U_restored)
        
        print("--- Matrix Normalization and Fidelity Test ---")
        print(f"Noise Intensity:    {noise_intensity}")
        print(f"Fidelity:           {fid:.5f}")
        print(f"Amplitude Fidelity: {fid_amp:.5f}")
        
        # 6. Test make_real_border
        R, Dleft, Dright = make_real_border(U_target)

        # Calculate the sum of the absolute imaginary parts on the first row and column
        # If the function works correctly, these values should be extremely close to 0
        imag_sum_row = np.sum(np.abs(np.imag(R[0, :])))
        imag_sum_col = np.sum(np.abs(np.imag(R[:, 0])))
        
        print("\n--- Real Border Test ---")
        print(f"Residual imaginary sum (1st row): {imag_sum_row:.2e}")
        print(f"Residual imaginary sum (1st col): {imag_sum_col:.2e}")
        
    except Exception as e:
        print(f"An error occurred: {e}.")