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
from __future__ import annotations

import numpy as np
import torch


def sinkhorn(U: torch.Tensor, tol: float = 1e-6, max_iter: int = 1000) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize a (batch of) positive matrices using the Sinkhorn–Knopp algorithm in PyTorch.

    The algorithm iteratively rescales rows and columns to make each matrix
    doubly stochastic (rows and columns sum to 1), returning:
        U_input = D_out_mat @ U_norm @ D_in_mat

    Parameters
    ----------
    U : torch.Tensor, shape (N, N) or (B, N, N)
        Input matrix or batch of matrices with strictly positive real entries.
    tol : float, optional
        Convergence tolerance on row/col sums. Default is 1e-6.
    max_iter : int, optional
        Maximum number of iterations allowed. Default is 1000.

    Returns
    -------
    U_norm : torch.Tensor
        Doubly stochastic matrix (or batch).
    D_in : torch.Tensor
        Right diagonal scaling factors, shape (B, N) or (N,).
    D_out : torch.Tensor
        Left diagonal scaling factors, shape (B, N) or (N,).
    """

    # Promote single matrix to batch dimension if 2D
    U_work = U.to(dtype=torch.complex128, copy=True)

    B, N, _ = U_work.shape
    D_in = torch.ones((B, N), dtype=U_work.dtype, device=U_work.device)
    D_out = torch.ones((B, N), dtype=U_work.dtype, device=U_work.device)

    for i in range(max_iter):
        # Column normalization
        col_sums = torch.sum(U_work, axis=1)
        T_in = 1.0 / col_sums
        U_work = U_work * T_in.unsqueeze(1)
        D_in *= T_in

        # Row normalization
        row_sums = torch.sum(U_work, axis=2)
        T_out = 1.0 / row_sums
        U_work = T_out.unsqueeze(2) * U_work
        D_out *= T_out

        # Check convergence
        err_col = torch.max(torch.abs(1.0 - col_sums)).item()
        err_row = torch.max(torch.abs(1.0 - row_sums)).item()
        if max(err_col, err_row) < tol:
            break

    else:
        print(f"[WARNING] Sinkhorn did not converge within {max_iter} iterations.")

    return U_work, D_in, D_out


def amplitude_fidelity(U_target: torch.Tensor, U_exp: torch.Tensor) -> torch.Tensor :
    """
    Calculate the amplitude fidelity between two complex matrices.

    Evaluates similarity between element magnitudes, ignoring phase
    differences:
        F_amp = sum(|U_target| * |U_exp|) / N

    Parameters
    ----------
    U_target : torch.Tensor, shape (N, N) or (B, N, N)
        The target complex matrix (or batch).
    U_exp : torch.Tensor, shape (N, N) or (B, N, N)
        The experimental complex matrix (or batch).

    Returns
    -------
    torch.Tensor or float
        Amplitude fidelity values in [0, 1]. Returns a 1D Tensor of shape (B,).
    """
    return fidelity(torch.abs(U_target), torch.abs(U_exp))


def fidelity(U_target: torch.Tensor, U_exp: torch.Tensor) -> torch.Tensor | float:
    """Calculate the fidelity between two complex unitary matrices.

    Fidelity is defined as:
        F = |Tr(U_target^H @ U_exp)| / N

    Parameters
    ----------
    U_target : torch.Tensor, shape (N, N) or (B, N, N)
        The ideal or target unitary matrix (or batch).
    U_exp : torch.Tensor, shape (N, N) or (B, N, N)
        The experimental or reconstructed matrix (or batch).

    Returns
    -------
    torch.Tensor or float
        Fidelity value(s) in [0, 1]. Returns a float for 2D inputs,
        or a 1D Tensor of shape (B,) for batch inputs.
    """

    N = U_target.shape[-1]

    # Calculate overlap using trace of (U_target^H @ U_exp)
    overlap = torch.sum(U_target.conj() * U_exp, dim=(-2, -1))

    # Calculate fidelity
    fid = torch.abs(overlap) / N

    return fid


def make_real_border(U: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Make a unitary matrix or batch of matrices "real-bordered".

    Calculates the diagonal unitaries D_left and D_right such that R = D_left @ U
    @ D_right has entirely real elements on its first row and first column.

    Parameters
    ----------
    U : torch.Tensor, shape (B, N, N) or (N, N)
        The input unitary matrix (or batch) to be modified.

    Returns
    -------
    R : torch.Tensor
        The real-bordered unitary matrix or batch.
    D_left_mat : torch.Tensor
        The left diagonal unitary correction matrix or batch, shape (B, N, N).
    D_right_mat : torch.Tensor
        The right diagonal unitary correction matrix or batch, shape (B, N, N).
    """
    B, N, _ = U.shape

    # D_right calculation: first row phase cancellation
    # Extract phases of the first row: (B, N)
    row_phases = torch.angle(U[:, 0, :])
    D_right = torch.exp(-1j * row_phases)  # (B, N)

    # Apply D_right: (B, N, N) @ (B, 1, N)
    U1 = U * D_right.unsqueeze(-2)

    # D_left calculation: first column phase cancellation
    # Extract phases of the first column: (B, N)
    col_phases = torch.angle(U1[:, :, 0])
    D_left = torch.exp(-1j * col_phases)  # (B, N)

    # Apply D_left: (B, N, N) @ (B, N, 1)
    R = U1 * D_left.unsqueeze(-1)

    # Return as diagonal matrices
    D_left_mat = torch.diag_embed(D_left)
    D_right_mat = torch.diag_embed(D_right)

    return R, D_left_mat, D_right_mat


def get_haar_unitary(size: int, batch_size: int = 1, device: torch.device | str = "cpu") -> torch.Tensor:
    """
    Generate a batch of Haar-distributed unitary matrices using QR decomposition.
    This avoids Python loops by leveraging vectorized QR decomposition.

    Parameters
    ----------
    size : int
        The dimension N of the NxN unitary matrix.
    batch_size : int, optional
        The number of matrices to generate. Defaults to 1.

    Returns
    -------
    torch.Tensor
        A complex NxN Haar-random unitary matrix (shape (N, N) if batch_size=1,
        or shape (B, N, N) if batch_size > 1).
    """
    # Generate random complex matrices: (B, N, N).
    # Using standard normal distribution for real and imaginary parts
    real = torch.randn(batch_size, size, size, dtype=torch.float64, device=device)
    imag = torch.randn(batch_size, size, size, dtype=torch.float64, device=device)
    X = torch.complex(real, imag) / (2.0**0.5)

    # QR decomposition (vectorized over the first dimension)
    Q, R = torch.linalg.qr(X)

    # Adjust for Haar measure.
    # The QR decomposition gives a unitary Q, but the sign/phase of the
    # columns might be arbitrary. We multiply by the phase of the diagonal of R
    # to ensure the distribution is uniquely Haar.
    diag_R = torch.diagonal(R, dim1=-2, dim2=-1)
    phases = diag_R / torch.abs(diag_R)

    # Apply phases: Q @ diag(phases)
    Q *= phases.unsqueeze(-2)

    return Q.to(dtype=torch.complex128)



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
    U_test = U_batch

    # 1. Test Unitarity: U^H @ U should be I
    U_H = U_test.adjoint()
    identity_batch = torch.eye(size, dtype=U_test.dtype, device=U_test.device)
    unitarity_diff = torch.bmm(U_H, U_test) - identity_batch 
    error_unitarity = torch.mean(torch.abs(unitarity_diff)).item()
    print(f"Mean Unitarity Error:      {error_unitarity:.2e}")

    # 2. Test Eigenvalue Distribution:
    # Calculate eigenvalues for the whole batch
    eigenvalues = torch.linalg.eigvals(U_batch).flatten()

    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot the unit circle reference
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.3, label="Unit Circle")

    # Plot the eigenvalues
    ax.scatter(eigenvalues.real, eigenvalues.imag, s=1, alpha=0.2, c="blue", label="Eigenvalues")
    ax.set_aspect("equal")
    ax.set_xlabel("Real Part")
    ax.set_ylabel("Imaginary Part")
    ax.set_title(f"Eigenvalue Distribution (Haar Unitary, N={U_test.shape[-1]})")
    ax.legend()
    plt.show()

    # OTHER TOOL VALIDATION
    print(f"\n--- Matrix Normalization and Fidelity Test (N={size}, Batch Size: {batch_size}) ---")

    # 1. Create target random Haar matrices
    U_target = get_haar_unitary(size, batch_size=batch_size)

    # 2. Add noise
    # Maintain the dimensional symmetry of the noise based on the target.
    noise = (torch.randn(batch_size, size, size) + 1j * torch.randn(batch_size, size, size)) * noise_intensity
    U_noisy = U_target + noise

    # 3. Use Sinkhorn to recover a doubly stochastic matrix
    U_power = torch.abs(U_noisy) ** 2
    U_norm_power, _, _ = sinkhorn(U_power)

    # Dimensional promotion for safe row/col sums
    U_norm_test = U_norm_power
    max_row_err = torch.max(torch.abs(torch.sum(U_norm_test, axis=2) - 1.0))
    max_col_err = torch.max(torch.abs(torch.sum(U_norm_test, axis=1) - 1.0))
    print(f"Sinkhorn Max Row Error:    {max_row_err:.2e}")
    print(f"Sinkhorn Max Col Error:    {max_col_err:.2e}")

    # 4. Reconstruct normalized complex matrix
    # Broadcasting phase preservation across the batch
    eps = 1e-12
    phase_factors = U_noisy / (torch.abs(U_noisy) + eps)
    U_restored = torch.sqrt(U_norm_power) * phase_factors

    # 5. Calculate fidelity metrics
    # The method fidelity returns an array of shape (B,) or a scalar
    fids = fidelity(U_target, U_restored)
    fids_amp = amplitude_fidelity(U_target, U_restored)

    print(f"Noise Intensity:           {noise_intensity}")
    print(f"Mean Fidelity:             {torch.mean(fids):.5f}")
    print(f"Mean Amp Fidelity:         {torch.mean(fids_amp):.5f}")

    # 6. Test make_real_border
    # R is (B, N, N) or (N, N)
    R, _, _ = make_real_border(U_target)
    R_test = R
    
    # Calculate the max sum of absolute imaginary parts on the first row and column
    # Sum over dimensions 1 or 2 (within each matrix), then extract max over the batch
    imag_sum_row = torch.max(
        torch.sum(torch.abs(torch.imag(R_test[:, 0, :])), dim=1)
    ).item()
    imag_sum_col = torch.max(
        torch.sum(torch.abs(torch.imag(R_test[:, :, 0])), dim=1)
    ).item()

    print("\n--- Real Border Test (Batch Average) ---")
    print(f"Residual imaginary sum (1st row): {imag_sum_row:.2e}")
    print(f"Residual imaginary sum (1st col): {imag_sum_col:.2e}")
