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
get_haar_unitary(size, batch_size=1)
    Generate a complex random Haar-distributed unitary matrix or batch.
"""

import numpy as np
from typing import Tuple


def sinkhorn(U: np.ndarray, tol: float = 1e-6, max_iter: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize a (batch of) positive matrices using the Sinkhorn–Knopp algorithm.

    The algorithm iteratively rescales the rows and columns of each matrix
    to make it approximately doubly stochastic (rows and columns sum to 1).
    It returns both the normalized matrix and the diagonal correction
    factors such that:

        U = D_out @ U_norm @ D_in

    Parameters
    ----------
    U : np.ndarray, shape (N, N) or (B, N, N)
        Input matrix or batch of matrices with strictly positive entries.
    tol : float, optional
        Convergence tolerance. Default is 1e-6.
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
    
    B, N, _ = U_work.shape
    D_in = np.ones((B, N))
    D_out = np.ones((B, N))
    
    for i in range(max_iter):
        # Column normalization
        col_sums = np.sum(U_work, axis=1)
        T_in = 1.0 / col_sums
        U_work = U_work * T_in[:, np.newaxis, :]
        D_in *= T_in
        
        # Row normalization
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
    (B, N, N) matrices.
    
    Parameters
    ----------
    U_target : np.ndarray, shape (N, N) or (B, N, N)
        The ideal or target unitary matrix (or batch of matrices).
    U_exp : np.ndarray, shape (N, N) or (B, N, N)
        The experimental or measured matrix (or batch of matrices).

    Returns
    -------
    np.ndarray
        Amplitude fidelity values between 0 and 1 (scalar or array of shape (B,)).
    """
    return fidelity(np.abs(U_target), np.abs(U_exp))


def fidelity(U_target: np.ndarray, U_exp: np.ndarray) -> np.ndarray:
    """
    Calculate the fidelity between two complex unitary matrices.

    The fidelity is defined as the normalized absolute trace of the 
    overlap between the target and experimental matrices:
        
        F = |Tr(U_target^H @ U_exp)| / N.
    
    This metric is sensitive to relative phases but invariant to a 
    global phase shift. Supports both single (N, N) and batch (B, N, N) 
    matrices.
    
    Parameters
    ----------
    U_target : np.ndarray, shape (N, N) or (B, N, N)
        The ideal or target unitary matrix (or batch of matrices).
    U_exp : np.ndarray, shape (N, N) or (B, N, N)
        The experimental or measured matrix (or batch of matrices).

    Returns
    -------
    np.ndarray
        Fidelity values between 0 and 1 (scalar or array of shape (B,)).
    """
    # Check dimension to guarantee vectorial shape
    is_2d = U_target.ndim == 2
    if is_2d:
        U_target = U_target[np.newaxis, ...]    
    if U_exp.ndim == 2:
        U_exp = U_exp[np.newaxis, ...]
    
    N = U_target.shape[-1]
    
    # Calculate overlap using trace of (U_target^H @ U_exp)
    overlap = np.sum(U_target.conj() * U_exp, axis=(-2, -1))
    
    # Calculate fidelity
    fid = np.abs(overlap) / N
    
    return fid[0] if is_2d else fid


def make_real_border(U: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Make a unitary matrix or batch of matrices "real-bordered".

    Calculates the diagonal unitaries D_left and D_right such that
    R = D_left @ U @ D_right has entirely real elements on its first row 
    and first column. The elements of the first row and first column 
    of the input matrix U must be non-vanishing.

    Parameters
    ----------
    U : np.ndarray, shape (N, N) or (B, N, N)
        The input unitary matrix (or batch) to be modified.

    Returns
    -------
    R : np.ndarray
        The real-bordered unitary matrix (or batch).
    D_left : np.ndarray
        The left diagonal unitary correction matrix (or batch).
    D_right : np.ndarray
        The right diagonal unitary correction matrix (or batch).
    """
    # Check dimension to guarantee vectorial shape
    is_2d = U.ndim == 2
    if is_2d:
        U = U[np.newaxis, ...]
        
    B, N, _ = U.shape
    
    # D_right calculation: first row phase cancellation
    # Extract phases of the first row: (B, N)
    row_phases = np.angle(U[:, 0, :])
    D_right = np.exp(-1j * row_phases) # (B, N)
    
    # Apply D_right: (B, N, N) @ (B, N, N_diag)
    U1 = U * D_right[:, np.newaxis, :]
    
    # D_left calculation: first column phase cancellation
    # Extract phases of the first column: (B, N)
    col_phases = np.angle(U1[:, :, 0])
    D_left = np.exp(-1j * col_phases) # (B, N)
    
    # Apply D_left: (B, N_diag, N) @ (B, N, N)
    R = U1 * D_left[:, :, np.newaxis]
    
    if is_2d:
        return R[0], np.diag(D_left[0]), np.diag(D_right[0])
    
    # Return as diagonal matrices for consistency with previous 2D implementation
    # Note: Using batch-wise diagonal creation
    D_left_mat = np.eye(N) * D_left[:, np.newaxis, :]
    D_right_mat = np.eye(N) * D_right[:, np.newaxis, :]
    
    return R, D_left_mat, D_right_mat


def get_haar_unitary(size: int, batch_size: int = 1) -> np.ndarray:
    """
    Generate a batch of Haar-distributed unitary matrices using QR decomposition.
    This avoids Python loops by leveraging NumPy/SciPy's vectorized QR.

    Parameters
    ----------
    size : int
        The dimension N of the NxN unitary matrix.
    batch_size : int, optional
        The number of matrices to generate. Defaults to 1.

    Returns
    -------
    np.ndarray
        A complex NxN Haar-random unitary matrix (shape (N, N) if batch_size=1, 
        or shape (B, N, N) if batch_size > 1).
    """
    # Generate random complex matrices: (B, N, N).
    # Using standard normal distribution for real and imaginary parts
    X = (np.random.randn(batch_size, size, size) + 1j * np.random.randn(batch_size, size, size)) / np.sqrt(2)
    
    # QR decomposition (vectorized over the first dimension)
    # np.linalg.qr returns Q and R. Q is unitary.
    Q, R = np.linalg.qr(X)
    
    # Adjust for Haar measure.
    # The QR decomposition gives a unitary Q, but the sign/phase of the 
    # columns might be arbitrary. We multiply by the phase of the diagonal of R 
    # to ensure the distribution is uniquely Haar.
    phases = np.diagonal(R, axis1=1, axis2=2)
    phases = phases / np.abs(phases)    
    
    # Apply phases: Q @ diag(phases)
    Q *= phases[:, np.newaxis, :]
    
    return Q[0] if batch_size == 1 else Q
    
    
# ==================================================================
# Entry Point
# ==================================================================
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt   
    
    # Configuration
    size = 24
    batch_size = 100
    noise_intensity = 0.1

    # HAAR VALIDATION
    print(f"--- Statistical Verification of Haar Distribution (N={size}, Batch Size: {batch_size}) ---")
    U_batch = get_haar_unitary(size, batch_size)
    
    # Dimensional promotion for 3D testing to avoid shape errors
    U_test = U_batch[np.newaxis, ...] if U_batch.ndim == 2 else U_batch
    
    # 1. Test Unitarity: U^H @ U should be I
    U_H = U_test.conj().transpose(0, 2, 1)
    error_unitarity = np.mean(np.abs(U_H @ U_test - np.eye(size)))
    print(f"Mean Unitarity Error:      {error_unitarity:.2e}")
    
    # 2. Test Eigenvalue Distribution: 
    # Calculate eigenvalues for the whole batch
    eigenvalues = np.linalg.eigvals(U_batch).flatten()
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot the unit circle reference
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, label='Unit Circle')
    
    # Plot the eigenvalues
    ax.scatter(eigenvalues.real, eigenvalues.imag, s=1, alpha=0.2, c='blue', label='Eigenvalues')
    
    ax.set_aspect('equal')
    ax.set_xlabel('Real Part')
    ax.set_ylabel('Imaginary Part')
    ax.set_title(f'Eigenvalue Distribution (Haar Unitary, N={U_test.shape[-1]})')
    ax.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()
    
    # OTHER TOOL VALIDATION   
    print(f"\n--- Matrix Normalization and Fidelity Test (N={size}, Batch Size: {batch_size}) ---")

    # 1. Create target random Haar matrices
    U_target = get_haar_unitary(size, batch_size=batch_size)
    
    # 2. Add noise
    # Maintain the dimensional symmetry of the noise based on the target.
    if U_target.ndim == 2:
        noise = (np.random.randn(size, size) + 1j * np.random.randn(size, size)) * noise_intensity
    else:
        noise = (np.random.randn(batch_size, size, size) + 1j * np.random.randn(batch_size, size, size)) * noise_intensity
        
    U_noisy = U_target + noise
    
    # 3. Use Sinkhorn to recover a doubly stochastic matrix
    U_power = np.abs(U_noisy)**2
    U_norm_power, _, _ = sinkhorn(U_power)
    
    # Dimensional promotion for safe row/col sums
    U_norm_test = U_norm_power[np.newaxis, ...] if U_norm_power.ndim == 2 else U_norm_power
    max_row_err = np.max(np.abs(np.sum(U_norm_test, axis=2) - 1.0))
    max_col_err = np.max(np.abs(np.sum(U_norm_test, axis=1) - 1.0))
    print(f"Sinkhorn Max Row Error:    {max_row_err:.2e}")
    print(f"Sinkhorn Max Col Error:    {max_col_err:.2e}")
    
    # 4. Reconstruct normalized complex matrix
    # Broadcasting phase preservation across the batch
    U_restored = np.sqrt(U_norm_power) * np.exp(1j * np.angle(U_noisy))
    
    # 5. Calculate fidelity metrics
    # The method fidelity returns an array of shape (B,) or a scalar
    fids = fidelity(U_target, U_restored)
    fids_amp = amplitude_fidelity(U_target, U_restored)
    
    print(f"Noise Intensity:           {noise_intensity}")
    print(f"Mean Fidelity:             {np.mean(fids):.5f}")
    print(f"Mean Amp Fidelity:         {np.mean(fids_amp):.5f}")
    
    # 6. Test make_real_border
    # R is (B, N, N) or (N, N)
    R, _, _ = make_real_border(U_target)

    # Dimensional promotion for 3D testing
    R_test = R[np.newaxis, ...] if R.ndim == 2 else R

    # Calculate the max sum of absolute imaginary parts on the first row and column
    # Sum over dimensions 1 or 2 (within each matrix), then extract max over the batch
    imag_sum_row = np.max(np.sum(np.abs(np.imag(R_test[:, 0, :])), axis=1))
    imag_sum_col = np.max(np.sum(np.abs(np.imag(R_test[:, :, 0])), axis=1))
    
    print("\n--- Real Border Test (Batch Average) ---")
    print(f"Residual imaginary sum (1st row): {imag_sum_row:.2e}")
    print(f"Residual imaginary sum (1st col): {imag_sum_col:.2e}")