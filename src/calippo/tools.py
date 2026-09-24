#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 12:07:46 2025

@author: Francesco Ceccarelli, Andrea Crespi

Utilities and helpers for UPP calibration/operation.

This module provides utility functions for measuring and processing unitaries
on a Universal Photonic Processor (UPP), including Sinkhorn normalization,
fidelity metrics, random unitary generation and other. PyTorch tensor-based
implementation supporting batched operations.

Functions
---------
sinkhorn(U, tol=1e-3, max_iter=1000)
    Normalize tensors to be doubly stochastic using Sinkhorn-Knopp.
amplitude_fidelity(U_target, U_exp)
    Calculate fidelity based on the magnitude of tensor elements.
fidelity(U_target, U_exp)
    Calculate the standard fidelity for complex unitaries.
make_real_border(U)
    Make a unitary tensor "real-bordered" by zeroing angles on 1st row/col.
get_haar_unitary(size, batch_size=1)
    Generate a complex random Haar-distributed unitary tensor or batch.
"""

import numpy as np
import torch
from typing import Tuple, Optional


def sinkhorn(U: torch.Tensor, tol: float = 1e-6, max_iter: int = 1000) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize a (batch of) positive tensors using the Sinkhorn–Knopp algorithm.

    The algorithm iteratively rescales the rows and columns of each tensor
    to make it approximately doubly stochastic (rows and columns sum to 1).
    It returns both the normalized tensor and the diagonal correction
    factors such that:

        U = D_out @ U_norm @ D_in

    Parameters
    ----------
    U : torch.Tensor, shape (N, N) or (B, N, N)
        Input tensor or batch of tensors with strictly positive entries.
    tol : float, optional
        Convergence tolerance. Default is 1e-6.
    max_iter : int, optional
        Maximum number of iterations allowed. Default is 1000.

    Returns
    -------
    U_norm : torch.Tensor
        Normalized tensor (or batch), approximately doubly stochastic.
    D_in : torch.Tensor
        Right diagonal scaling factors.
    D_out : torch.Tensor
        Left diagonal scaling factors.
    """
    # Promote single tensor to batch dimension if 2D
    is_2d = U.ndim == 2
    if is_2d:
        U_work = U.unsqueeze(0).clone().double()
    else:
        U_work = U.clone().double()

    B, N, _ = U_work.shape
    device = U.device
    
    D_in = torch.ones((B, N), dtype=torch.float64, device=device)
    D_out = torch.ones((B, N), dtype=torch.float64, device=device)

    for i in range(max_iter):
        # Column normalization
        col_sums = torch.sum(U_work, dim=1)
        T_in = 1.0 / col_sums
        U_work = U_work * T_in.unsqueeze(1)
        D_in *= T_in

        # Row normalization
        row_sums = torch.sum(U_work, dim=2)
        T_out = 1.0 / row_sums
        U_work = T_out.unsqueeze(2) * U_work
        D_out *= T_out

        # Check convergence
        err_col = torch.max(torch.abs(1 - col_sums))
        err_row = torch.max(torch.abs(1 - row_sums))
        if max(err_col.item(), err_row.item()) < tol:
            break

    else:
        print(f"[WARNING] Sinkhorn did not converge within {max_iter} iterations.")

    if is_2d:
        return U_work[0].to(U.dtype), D_in[0].to(U.dtype), D_out[0].to(U.dtype)

    return U_work.to(U.dtype), D_in.to(U.dtype), D_out.to(U.dtype)


def amplitude_fidelity(U_target: torch.Tensor, U_exp: torch.Tensor) -> torch.Tensor:
    """
    Calculate the amplitude fidelity between two complex tensors.

    This metric evaluates the similarity between the magnitudes of the
    tensor elements, ignoring phase differences. It is useful when only
    power measurements are available. Supports single (N, N) or batch
    (B, N, N) tensors.

    Parameters
    ----------
    U_target : torch.Tensor, shape (N, N) or (B, N, N)
        The ideal or target unitary tensor (or batch of tensors).
    U_exp : torch.Tensor, shape (N, N) or (B, N, N)
        The experimental or measured tensor (or batch of tensors).

    Returns
    -------
    torch.Tensor
        Amplitude fidelity values between 0 and 1 (scalar or tensor of shape (B,)).
    """
    return fidelity(torch.abs(U_target), torch.abs(U_exp))


def fidelity(U_target: torch.Tensor, U_exp: torch.Tensor) -> torch.Tensor:
    """
    Calculate the fidelity between two complex unitary tensors.

    The fidelity is defined as the normalized absolute trace of the
    overlap between the target and experimental tensors:

        F = |Tr(U_target^H @ U_exp)| / N.

    This metric is sensitive to relative phases but invariant to a
    global phase shift. Supports both single (N, N) and batch (B, N, N)
    tensors.

    Parameters
    ----------
    U_target : torch.Tensor, shape (N, N) or (B, N, N)
        The ideal or target unitary tensor (or batch of tensors).
    U_exp : torch.Tensor, shape (N, N) or (B, N, N)
        The experimental or measured tensor (or batch of tensors).

    Returns
    -------
    torch.Tensor
        Fidelity values between 0 and 1 (scalar or tensor of shape (B,)).
    """
    # Check dimension to guarantee vectorial shape
    is_2d = U_target.ndim == 2
    if is_2d:
        U_target = U_target.unsqueeze(0)
    if U_exp.ndim == 2:
        U_exp = U_exp.unsqueeze(0)

    N = U_target.shape[-1]

    # Calculate overlap using trace of (U_target^H @ U_exp)
    overlap = torch.sum(U_target.conj() * U_exp, dim=(-2, -1))

    # Calculate fidelity
    fid = torch.abs(overlap) / N

    return fid[0] if is_2d else fid


def make_real_border(U: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Make a unitary tensor or batch of tensors "real-bordered".

    Calculates the diagonal unitaries D_left and D_right such that
    R = D_left @ U @ D_right has entirely real elements on its first row
    and first column. The elements of the first row and first column
    of the input tensor U must be non-vanishing.

    Parameters
    ----------
    U : torch.Tensor, shape (N, N) or (B, N, N)
        The input unitary tensor (or batch) to be modified.

    Returns
    -------
    R : torch.Tensor
        The real-bordered unitary tensor (or batch).
    D_left : torch.Tensor
        The left diagonal unitary correction tensor (or batch).
    D_right : torch.Tensor
        The right diagonal unitary correction tensor (or batch).
    """
    # Check dimension to guarantee vectorial shape
    is_2d = U.ndim == 2
    if is_2d:
        U = U.unsqueeze(0)

    B, N, _ = U.shape

    # D_right calculation: first row phase cancellation
    # Extract phases of the first row: (B, N)
    row_phases = torch.angle(U[:, 0, :])
    D_right = torch.exp(-1j * row_phases)  # (B, N)

    # Apply D_right: (B, N, N) @ (B, N, N_diag)
    U1 = U * D_right.unsqueeze(1)

    # D_left calculation: first column phase cancellation
    # Extract phases of the first column: (B, N)
    col_phases = torch.angle(U1[:, :, 0])
    D_left = torch.exp(-1j * col_phases)  # (B, N)

    # Apply D_left: (B, N_diag, N) @ (B, N, N)
    R = U1 * D_left.unsqueeze(2)

    if is_2d:
        return R[0], torch.diag(D_left[0]), torch.diag(D_right[0])

    # Return as diagonal matrices for consistency with previous 2D implementation
    # Note: Using torch.diag_embed for batch-wise diagonal creation
    D_left_mat = torch.diag_embed(D_left)
    D_right_mat = torch.diag_embed(D_right)

    return R, D_left_mat, D_right_mat


def get_haar_unitary(size: int, batch_size: int = 1, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Generate a batch of Haar-distributed unitary tensors using QR decomposition.
    This avoids Python loops by leveraging PyTorch's vectorized QR.

    Parameters
    ----------
    size : int
        The dimension N of the NxN unitary tensor.
    batch_size : int, optional
        The number of tensors to generate. Defaults to 1.
    device: torch.device, optional
        The device to initialize the tensors on.

    Returns
    -------
    torch.Tensor
        A complex NxN Haar-random unitary tensor (shape (N, N) if batch_size=1,
        or shape (B, N, N) if batch_size > 1).
    """
    if device is None:
        device = torch.device("cpu")
        
    # Generate random complex tensors: (B, N, N).
    # Using standard normal distribution for real and imaginary parts
    X = (torch.randn(batch_size, size, size, dtype=torch.float64, device=device) + 
         1j * torch.randn(batch_size, size, size, dtype=torch.float64, device=device)) / np.sqrt(2)

    # QR decomposition (vectorized over the first dimension)
    # torch.linalg.qr returns Q and R. Q is unitary.
    Q, R = torch.linalg.qr(X)

    # Adjust for Haar measure.
    # The QR decomposition gives a unitary Q, but the sign/phase of the
    # columns might be arbitrary. We multiply by the phase of the diagonal of R
    # to ensure the distribution is uniquely Haar.
    phases = torch.diagonal(R, dim1=1, dim2=2)
    phases = phases / torch.abs(phases)

    # Apply phases: Q @ diag(phases)
    Q *= phases.unsqueeze(1)

    return Q
    # return Q[0] if batch_size == 1 else Q


# ==================================================================
# Entry Point
# ==================================================================
if __name__ == "__main__":

    import matplotlib.pyplot as plt

    try:
        # Configuration
        size = 24
        batch_size = 100
        noise_intensity = 0.1

        # HAAR VALIDATION
        print(f"--- Statistical Verification of Haar Distribution (N={size}, Batch Size: {batch_size}) ---")
        U_batch = get_haar_unitary(size, batch_size)

        # Dimensional promotion for 3D testing to avoid shape errors
        U_test = U_batch.unsqueeze(0) if U_batch.ndim == 2 else U_batch

        # 1. Test Unitarity: U^H @ U should be I
        U_H = torch.adjoint(U_test)
        error_unitarity = torch.mean(torch.abs(torch.matmul(U_H, U_test) - torch.eye(size, dtype=U_test.dtype, device=U_test.device))).item()
        print(f"Mean Unitarity Error:      {error_unitarity:.2e}")

        # 2. Test Eigenvalue Distribution:
        # Calculate eigenvalues for the whole batch
        eigenvalues = torch.linalg.eigvals(U_batch).flatten().cpu().numpy()

        fig, ax = plt.subplots(figsize=(6, 6))

        # Plot the unit circle reference
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.3, label="Unit Circle")

        # Plot the eigenvalues
        ax.scatter(
            eigenvalues.real,
            eigenvalues.imag,
            s=1,
            alpha=0.2,
            c="blue",
            label="Eigenvalues",
        )

        ax.set_aspect("equal")
        ax.set_xlabel("Real Part")
        ax.set_ylabel("Imaginary Part")
        ax.set_title(f"Eigenvalue Distribution (Haar Unitary, N={U_test.shape[-1]})")
        ax.legend()
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.show()

        # OTHER TOOL VALIDATION
        print(f"\n--- Matrix Normalization and Fidelity Test (N={size}, Batch Size: {batch_size}) ---")

        # 1. Create target random Haar matrices
        U_target = get_haar_unitary(size, batch_size=batch_size)

        # 2. Add noise
        # Maintain the dimensional symmetry of the noise based on the target.
        if U_target.ndim == 2:
            noise = (torch.randn(size, size, dtype=torch.float64) + 1j * torch.randn(size, size, dtype=torch.float64)) * noise_intensity
        else:
            noise = (
                torch.randn(batch_size, size, size, dtype=torch.float64) + 1j * torch.randn(batch_size, size, size, dtype=torch.float64)
            ) * noise_intensity

        U_noisy = U_target + noise.to(U_target.device)

        # 3. Use Sinkhorn to recover a doubly stochastic matrix
        U_power = torch.abs(U_noisy) ** 2
        U_norm_power, _, _ = sinkhorn(U_power)

        # Dimensional promotion for safe row/col sums
        U_norm_test = U_norm_power.unsqueeze(0) if U_norm_power.ndim == 2 else U_norm_power
        max_row_err = torch.max(torch.abs(torch.sum(U_norm_test, dim=2) - 1.0)).item()
        max_col_err = torch.max(torch.abs(torch.sum(U_norm_test, dim=1) - 1.0)).item()
        print(f"Sinkhorn Max Row Error:    {max_row_err:.2e}")
        print(f"Sinkhorn Max Col Error:    {max_col_err:.2e}")

        # 4. Reconstruct normalized complex matrix
        # Broadcasting phase preservation across the batch
        U_restored = torch.sqrt(U_norm_power) * torch.exp(1j * torch.angle(U_noisy))

        # 5. Calculate fidelity metrics
        # The method fidelity returns a tensor of shape (B,) or a scalar
        fids = fidelity(U_target, U_restored)
        fids_amp = amplitude_fidelity(U_target, U_restored)

        print(f"Noise Intensity:           {noise_intensity}")
        print(f"Mean Fidelity:             {torch.mean(fids).item():.5f}")
        print(f"Mean Amp Fidelity:         {torch.mean(fids_amp).item():.5f}")

        # 6. Test make_real_border
        # R is (B, N, N) or (N, N)
        R, _, _ = make_real_border(U_target)

        # Dimensional promotion for 3D testing
        R_test = R.unsqueeze(0) if R.ndim == 2 else R

        # Calculate the max sum of absolute imaginary parts on the first row and column
        # Sum over dimensions 1 or 2 (within each matrix), then extract max over the batch
        imag_sum_row = torch.max(torch.sum(torch.abs(torch.imag(R_test[:, 0, :])), dim=1)).item()
        imag_sum_col = torch.max(torch.sum(torch.abs(torch.imag(R_test[:, :, 0])), dim=1)).item()

        print("\n--- Real Border Test (Batch Average) ---")
        print(f"Residual imaginary sum (1st row): {imag_sum_row:.2e}")
        print(f"Residual imaginary sum (1st col): {imag_sum_col:.2e}")

    except Exception as e:
        print(f"{e}")

    finally:
        pass