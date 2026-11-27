"""
Utilities and helpers for UPP calibration/operation.

This module provides utility functions for measuring and processing unitaries
on a Universal Photonic Processor (UPP), including Sinkhorn normalization,
fidelity metrics, random unitary generation and other. PyTorch tensor-based
implementation supporting batched operations.
"""

from __future__ import annotations

import math

import numpy as np
import torch


def get_device(device: str | torch.device | None = None) -> torch.device:
    """Select the compute device.

    Parameters
    ----------
    device : str or torch.device, optional
        Requested device (e.g. "cpu", "cuda", "cuda:1"). If None, CUDA is used
        when available, otherwise the CPU. If CUDA is explicitly requested but
        not available, a message is printed and the CPU is used instead.

    Returns
    -------
    torch.device
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available: falling back to CPU.")
        device = torch.device("cpu")
    return device


def sinkhorn(
    U: torch.Tensor, tol: float = 1e-6, max_iter: int = 1000
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    W = U.to(U.dtype, copy=True)
    B, N, _ = W.shape

    # accumulated scalings: U_norm = diag(row_acc) @ U @ diag(col_acc)
    row_acc = torch.ones(B, N, dtype=W.dtype, device=W.device)
    col_acc = torch.ones(B, N, dtype=W.dtype, device=W.device)

    converged = False
    for _ in range(max_iter):
        # column sums -> (B, N)
        t_col = 1.0 / W.sum(dim=1)
        W = W * t_col.unsqueeze(1)
        col_acc *= t_col

        # rows sums -> (B, N)
        t_row = 1.0 / W.sum(dim=2)
        W = W * t_row.unsqueeze(2)
        row_acc *= t_row

        # rows are exact after the row step; check column residual only.
        err = (1.0 - W.sum(dim=1)).abs().amax()  # single host sync
        if err.item() < tol:
            converged = True
            break

    if not converged:
        print(f"[WARNING] Sinkhorn did not converge within {max_iter} iterations.")

    W = W.to(U.dtype)
    D_out = (1.0 / row_acc).to(U.dtype)
    D_in = (1.0 / col_acc).to(U.dtype)

    return W, D_in, D_out


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


def make_real_border(U: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    # Right phases from the first row: multiplies columns.
    ph_right = U[:, 0, :] / U[:, 0, :].abs().clamp_min(torch.finfo(U.real.dtype).tiny)
    U1 = U * ph_right.conj().unsqueeze(1)

    # Left phases from the first column of U1: multiplies rows.
    ph_left = U1[:, :, 0] / U1[:, :, 0].abs().clamp_min(torch.finfo(U.real.dtype).tiny)
    R = U1 * ph_left.conj().unsqueeze(2)

    # Factors satisfying U = D_left @ R @ D_right.
    D_left = torch.diag_embed(ph_left)
    D_right = torch.diag_embed(ph_right)
    return R, D_left, D_right


def get_haar_unitary(
    size: int,
    batch_size: int = 1,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Generate a batch of Haar-distributed unitary tensors using QR decomposition.
    This avoids Python loops by leveraging PyTorch's vectorized QR.

    Parameters
    ----------
    size : int
        The dimension N of the NxN unitary tensor.
    batch_size : int, optional
        The number of tensors to generate. Defaults to 1.
    device : torch.device, optional
        The device to initialize the tensors on.
    dtype : torch.dtype, optional
        The data type (real) used for the unitary. Deafult is torch.float64.
    seed : int, optional
        The seed for the generation of random Haar matrices.
        Default seed is None, no seed is applied, the unitaries changes at each function call.

    Returns
    -------
    torch.Tensor
        A complex NxN Haar-random unitary tensor (shape (N, N) if batch_size=1,
        or shape (B, N, N) if batch_size > 1).
    """
    device = get_device(device)

    gen = torch.Generator(device=device)
    if seed is None:
        gen.seed()
    else:
        gen.manual_seed(seed)

    # Complex Ginibre matrix: real and imaginary parts ~ N(0, 1/2).
    re, im = torch.randn(2, batch_size, size, size, generator=gen, dtype=dtype, device=device)
    X = torch.complex(re, im) / math.sqrt(2)

    Q, R = torch.linalg.qr(X)

    # Fix the column phases using diag(R) so the distribution is Haar.
    d = torch.diagonal(R, dim1=-2, dim2=-1)
    return Q * (d / d.abs()).unsqueeze(-2)


def clements_xtalk_sign(n_modes: int, dtype: torch.dtype = torch.int8) -> torch.Tensor:
    """
    Sign matrix of the thermal cross-talk of a Clements mesh with ``n_modes`` modes.

    Entry [i, j] is the sign of the phase change induced on shifter i (the victim) by the power dissipated in
    shifter j (the aggressor): +1, -1, or 0 when the two do not interact. The matrix is organized in quadrants
    (n = number of shifters / 2)::

        [ internal <- internal |        0             ]   top    (victims: internal shifters)
        [ external <- internal | external <- external ]   bottom (victims: external shifters)

    Which shifters interact
        * Internal victim: only the internal shifters of its own layer.
        * External victim: every shifter (internal or external) between its own layer and the layer where its
          light path, followed backwards, would hit the bottom waveguide. The path moves one waveguide down every
          two layers, so it reaches the bottom waveguide at layer ``victim.layer - 2 * (n_modes - 1 - victim.mode)``.
          The search stops at layer 2, where the unused input phase shifters sit.
        * Unused input shifters interact with nothing.

    Sign of an interaction
        A shifter heats the waveguides around it. The victim's phase is set by the heat on its own light path,
        so the aggressor counts as "+" when it lies on, or above, that path, and as "-" when it lies below.
        The path of an MZI shifter is traced backwards to the aggressor's layer, where it sits at waveguide
        ``victim.mode + (victim.layer - aggressor.layer) / 2``. Hence::

            sign = +1  if  aggressor.mode <= victim.mode + (victim.layer - aggressor.layer) / 2
                   -1  otherwise

        For aggressors in the victim's own layer this reduces to "+" for the shifter itself and for the ones
        below it, "-" for the ones above it.

    Parameters
    ----------
    n_modes : int
        Number of optical modes (>= 2).
    dtype : torch.dtype, optional
        Output dtype. Default is torch.int8; use torch.float64/torch.float32 to multiply directly with the model parameters.

    Returns
    -------
    torch.Tensor, shape (n_modes * (n_modes - 1), n_modes * (n_modes - 1))
        The sign matrix, with the phase ordering of ``clements_shifters``.
    """
    if not isinstance(n_modes, int) or n_modes < 2:
        raise ValueError(f"n_modes must be an integer >= 2, got {n_modes!r}.")

    # The first two columns of external phase shifters (N - 1 shifters in total) only set the phases of the input
    # modes: they do not affect the measured intensities, so they are not part of the cross-talk model.
    unused_external_cols = 2

    pairs = [(col, m) for col in range(n_modes) for m in range(col % 2, n_modes - 1, 2)]
    n = len(pairs)

    col, mode_idx = zip(*pairs)  # column and mode index of each MZI
    layer = torch.tensor([2 * c + 1 for c in col] + [2 * c for c in col])
    mode = torch.tensor(list(mode_idx) * 2)
    is_internal = torch.tensor([True] * n + [False] * n)
    used = torch.tensor([True] * n + [c >= unused_external_cols for c in col])

    # Rows are victims, columns are aggressors.
    victim_layer, aggressor_layer = layer[:, None], layer[None, :]
    victim_mode, aggressor_mode = mode[:, None], mode[None, :]

    # Who interacts with whom
    last_unused_layer = 2 * (unused_external_cols - 1)
    victim_to_bottom_layer = layer - 2 * (n_modes - 1 - mode)  # backward path from victim to bottom waveguide layer
    reach = torch.clamp(victim_to_bottom_layer, min=last_unused_layer)[:, None]  # earliest layer a victim feels

    internal_victim = is_internal[:, None] & is_internal[None, :] & (victim_layer == aggressor_layer)
    external_victim = ~is_internal[:, None] & (reach <= aggressor_layer) & (aggressor_layer <= victim_layer)
    interacts = (internal_victim | external_victim) & used[:, None] & used[None, :]

    # sign of the interaction (doubled to stay in integers: the path may sit between two waveguides)
    on_or_above_path = 2 * aggressor_mode <= 2 * victim_mode + (victim_layer - aggressor_layer)
    sign = torch.where(on_or_above_path, 1, -1)

    return (interacts * sign).to(dtype)


if __name__ == "__main__":

    import traceback

    import matplotlib.pyplot as plt

    try:
        # Configuration
        size = 24
        batch_size = 1000
        noise_intensity = 0.1

        # HAAR VALIDATION
        print(f"--- Statistical Verification of Haar Distribution (N={size}, Batch Size: {batch_size}) ---")
        U_batch = get_haar_unitary(size, batch_size)

        # Dimensional promotion for 3D testing to avoid shape errors
        U_test = U_batch.unsqueeze(0) if U_batch.ndim == 2 else U_batch

        # 1. Test Unitarity: U^H @ U should be I
        U_H = torch.adjoint(U_test)
        error_unitarity = torch.mean(
            torch.abs(torch.matmul(U_H, U_test) - torch.eye(size, dtype=U_test.dtype, device=U_test.device))
        ).item()
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
            noise = (
                torch.randn(size, size, dtype=torch.float64) + 1j * torch.randn(size, size, dtype=torch.float64)
            ) * noise_intensity
        else:
            noise = (
                torch.randn(batch_size, size, size, dtype=torch.float64)
                + 1j * torch.randn(batch_size, size, size, dtype=torch.float64)
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
        traceback.print_exc()
