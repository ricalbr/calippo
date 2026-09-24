#!/usr/bin/env python3

"""
Created on Thu Jun 11 16:07:49 2026

@author: Andrea Crespi, Francesco Ceccarelli, Riccardo Albiero

Digital twin models for Universal Photonic Processors (UPP).

This module defines objects implementing a photonic processor built
according to a sequence of mixing and phase layers. Multiple instances
are supported in parallel, namely multiple configurations of phases
can be calculated together using batch processing.

Classes
-------
Mesh
    Abstract base class defining the standard mesh for UPP digital twins.
Clements
    A photonic processor built according to Clements et al. Optica 3.12
    (2016): 1460-1465.
Bell
    A photonic processor built according to Bell et al. APL Photonics 6.7
    (2021).
"""

from __future__ import annotations

import dataclasses
import itertools
import pickle
import time
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy as sp
import torch
import torch.nn as nn

nparray = npt.NDArray


class Mesh(nn.Module, ABC):
    """
    Abstract base class for UPP meshes.
    It defines the common interface and internal structures that all mesh
    topologies must implement.

    Parameters
    ----------
    n_modes : int
        Number of optical modes (must be even in this version).
    calibration_file : str, optional
        Path to a pickle file containing the hardware calibration data.
        If None, a random calibration is generated.
    """

    def __init__(self, n_modes: int, calibration_file: str | None = None) -> None:
        super().__init__()

        if n_modes % 2 != 0:
            n_modes -= 1
            print("Modes must be EVEN for this version.\n", f"We shall take {n_modes} modes.")

        self.n_modes = n_modes  # Number of modes
        self.n_phases = (n_modes - 1) * n_modes  # Number of tunable parameters
        self.n_used_phases = (n_modes - 1) ** 2  # Number of in-use tunable parameters

        # This part builds the specific topology of the mesh (implemented by children)
        static_phases_mask_np, xtalk_mask_np = self._generate_masks()
        n_mixing_layers, M_np = self._generate_mixing_layers()
        self.n_mixing_layers = n_mixing_layers

        # Register as PyTorch buffers for device compatibility (non-trainable)
        # Masks that relate the used calibration parameters to the available ones
        self.register_buffer("static_phases_mask", torch.from_numpy(static_phases_mask_np).bool())
        self.register_buffer("xtalk_mask", torch.from_numpy(xtalk_mask_np).float())
        # Mode mixing implemented by directional couplers layers
        self.register_buffer("M", torch.from_numpy(M_np).to(torch.complex128))

        # This part populates xtalk (rad/mA2) and static phases (rad)
        if calibration_file is not None:
            sp_np, xt_np = self.import_calibration(calibration_file)
        else:
            sp_np, xt_np = self.generate_random_calibration(1.0, 2 * np.pi)

        # Register as Parameters to enable training (e.g., with Adam)
        self.static_phases = nn.Parameter(torch.from_numpy(sp_np).float())
        self.xtalk = nn.Parameter(torch.from_numpy(xt_np).float())

        # Output
        msg = (
            "A new processor was created.\n"
            + f"This has {self.n_modes} modes, {self.n_phases} tunable phases.\n"
            + f"Only {self.n_used_phases} phases are actually used.\n",
        )
        print(*msg)

    @abstractmethod
    def _generate_masks(self) -> tuple[nparray, nparray]:
        pass

    @abstractmethod
    def _generate_mixing_layers(self) -> tuple[int, nparray]:
        pass

    def import_calibration(self, fileName: str) -> tuple[nparray, nparray]:
        """
        Import calibration data from a pickle file.

        Parameters
        ----------
        fileName : str
            Path to the pickle file containing 'static_phases',
            'xt_matrix', and 'n_heaters'.

        Returns
        -------
        tuple of np.ndarray
            The static phases array (rad) and the cross-talk matrix (rad/mA2).

        Raises
        ------
        ValueError
            If the number of modes or phases in the calibration file
            does not match the processor architecture.
        """
        # Open file
        with open(fileName, "rb") as f:
            calibration = pickle.load(f)

        # Input validation
        if calibration["N"] != self.n_modes:
            raise ValueError(f"Calibration size mismatch: expected {self.n_modes} modes, " f"found {calibration['N']}.")

        if calibration["n_heaters"] != self.n_phases:
            raise ValueError(
                f"Calibration size mismatch: expected {self.n_phases} phases, " f"found {calibration['n_heaters']}."
            )

        # Fetch dataset
        static_phases = np.mod(calibration["static_phases"], 2 * np.pi)
        static_phases = static_phases * self.static_phases_mask.cpu().numpy()
        xtalk = calibration["xt_matrix"]
        xtalk = xtalk * self.xtalk_mask.cpu().numpy()

        return static_phases, xtalk

    def generate_random_calibration(
        self,
        xtalk_ampl: float,
        static_phases_ampl: float,
        noise: float | None = None,
    ) -> tuple[nparray, nparray]:
        """
        Generate a random calibration (static phases and cross-talk) for
        simulation purposes.

        Parameters
        ----------
        xtalk_ampl : float
            Average current-squared to phase gain (diagonal of the matrix) in rad/mA2.
        static_phases_ampl : float
            Amplitude for initial random static phase distribution in rad.
        noise : float, optional
            Noise amount on both diagonal and cross-talks. If None, defaults to xtalk_ampl/10.

        Returns
        -------
        tuple of np.ndarray
            An array of static phases (rad) and the cross-talk matrix (rad/mA2).
        """
        # Set noise level
        if noise is None:
            noise = xtalk_ampl / 10.0

        sp_mask_np = self.static_phases_mask.cpu().numpy()
        xt_mask_np = self.xtalk_mask.cpu().numpy()

        # Build dataset
        static_phases = np.mod(np.random.rand(self.n_phases) * static_phases_ampl, 2 * np.pi)
        static_phases = static_phases * sp_mask_np
        xtalk = np.eye(self.n_phases) * xtalk_ampl + (np.random.rand(self.n_phases, self.n_phases) - 0.5) * noise
        xtalk = xtalk * xt_mask_np

        return static_phases, xtalk

    def generate_random_currents(self, current_ampl: float, batch_size: int | None = None) -> torch.Tensor:
        """
        Generate random currents in mA. Supports single vectors or batches.

        Parameters
        ----------
        current_ampl : float
            Maximum current amplitude in mA.
        batch_size : int, optional
            If provided, generates a batch of shape (batch_size, n_phases).

        Returns
        -------
        torch.Tensor
            Tensor of randomized currents in mA.
        """
        # Check dimension in order to provide always squeezed shape
        if batch_size is None:
            I = torch.sqrt(torch.rand(self.n_phases)) * current_ampl
        else:
            I = torch.sqrt(torch.rand(batch_size, self.n_phases)) * current_ampl

        # Return only useful currents
        I *= self.static_phases_mask.cpu().float()
        return I

    def phases_from_currents(self, currents: torch.Tensor) -> torch.Tensor:
        """
        Calculate the optical phases for currents based on the cross-talk
        coefficients. Supports batches (B, n_phases) or single (n_phases,).

        Parameters
        ----------
        currents : torch.Tensor
            Tensor of currents in mA.

        Returns
        -------
        torch.Tensor
            Tensor of calculated phases in rad.

        Raises
        ------
        ValueError
            If the last dimension of the input array does not match the
            expected number of phases for the mesh.
        """
        # Check dimension to guarantee vectorial shape
        is_1d = currents.ndim == 1
        if is_1d:
            currents = currents.unsqueeze(0)

        # Input validation
        if currents.shape[-1] != self.n_phases:
            raise ValueError(
                f"Dimension mismatch: expected last dimension to be {self.n_phases}, found {currents.shape[-1]}."
            )

        # Dynamically cast internal parameters to match the input currents dtype
        static_phases = self.static_phases.to(currents.dtype)
        xtalk = self.xtalk.to(currents.dtype)

        # Calculate phases (Gradient support): (B, P) @ (P, P).T -> (B, P)
        # Now both are guaranteed to be the same precision
        phases = static_phases + (currents**2) @ xtalk.T

        # Keep 2*pi in the same dtype to prevent unexpected promotion
        pi_tensor = torch.tensor(2 * torch.pi, dtype=currents.dtype, device=currents.device)
        phases = phases % pi_tensor

        return phases.squeeze(0) if is_1d else phases

    @abstractmethod
    def U_from_phases(self, phases: torch.Tensor) -> torch.Tensor:
        pass

    def currents_from_phases(self, phases: torch.Tensor) -> torch.Tensor:
        """
        Calculate the currents in mA required to achieve target optical phases,
        compensating for cross-talk using the calibration matrix.
        Supports single or batch processing.

        Parameters
        ----------
        phases : torch.Tensor
            Tensor of target phase values in rad. Shape (n_phases,) or (B, n_phases).

        Returns
        -------
        torch.Tensor
            Tensor of currents in mA. Shape (n_phases,) or (B, n_phases).

        Raises
        ------
        ValueError
            If the last dimension of the input array does not match the
            expected number of phases for the mesh.
        """
        # Check dimension to guarantee vectorial shape
        is_1d = phases.ndim == 1
        if is_1d:
            phases = phases.unsqueeze(0)

        # Input validation
        if phases.shape[-1] != self.n_phases:
            raise ValueError(
                f"Dimension mismatch: expected last dimension to be {self.n_phases}, found {phases.shape[-1]}."
            )

        phases = phases % (2 * torch.pi)

        # Reduce dimensions to used parameters in order to avoid singularities
        phases_used = phases[:, self.static_phases_mask]

        # Cast static_phases to match the dtype of the input phases
        static_phases = self.static_phases[self.static_phases_mask].to(phases.dtype)

        used_rows = torch.any(self.xtalk_mask, dim=0)

        # Cast xtalk to match the dtype of the input phases
        xtalk = self.xtalk[used_rows][:, used_rows].to(phases.dtype)

        # Numerical inversion.
        # The equation is: xtalk @ I2^T = (phases_used - static_phases)^T
        # We transpose to align the batch axis (B) as the last dimension
        target_diff = (phases_used - static_phases).T
        I2 = torch.linalg.solve(xtalk, target_diff)

        # Keep 2*pi in the same dtype
        pi_tensor = torch.tensor(2 * torch.pi, dtype=phases.dtype, device=phases.device)

        while torch.any(I2 < 0):
            target_diff[I2 < 0] += pi_tensor
            I2 = torch.linalg.solve(xtalk, target_diff)

        # Transpose back and reconstruct original dimensions
        I2 = I2.T

        # Inherit the dtype directly from phases
        I = torch.zeros_like(phases)
        I[:, self.static_phases_mask] = torch.sqrt(I2)

        return I.squeeze(0) if is_1d else I

    @abstractmethod
    def phases_from_U(
        self,
        U_target: torch.Tensor,
        target_infidelity: float = 1e-6,
        **kwargs,
    ) -> torch.Tensor:
        pass


@dataclasses.dataclass
class MZIOp:
    """
    One batched MZI operation  W ← W @ T(m,n,θ,φ)^†.

    Attributes
    ----------
    m, n  : column indices of the 2×2 block  (m = n-1)
    theta : internal (splitting) phase, shape (B,)
    phi   : external phase, shape (B,)
    """

    m: int
    n: int
    theta: torch.Tensor
    phi: torch.Tensor

    def __iter__(self):
        return iter((self.m, self.n, self.theta, self.phi))


class Clements(Mesh):
    """
    A photonic processor built according to the Clements scheme.
    """

    def _generate_masks(self) -> tuple[nparray, nparray]:
        """
        Generate the boolean masks for the Clements architecture, specifying
        which phases and cross-talk elements are physically implemented.

        Returns
        -------
        tuple of np.ndarray
            A tuple containing (static_phases_mask, xtalk_mask).
        """
        # Generate static_phases_mask
        static_phases_mask = np.ones(self.n_phases, dtype=int)
        half_idx = self.n_phases // 2
        static_phases_mask[half_idx : half_idx + self.n_modes - 1] = 0
        static_phases_mask = static_phases_mask.astype(bool)

        # Initialize xtalk_mask
        xtalk_mask = np.zeros((self.n_phases, self.n_phases), dtype=float)

        # Calculate main blocks for the first M rows (Thetas)
        M = self.n_modes * (self.n_modes - 1) // 2

        B_sizes = []
        for i in range(self.n_modes):
            if i % 2 == 0:
                B_sizes.append(self.n_modes // 2)
            else:
                B_sizes.append(self.n_modes // 2 - 1)

        B_starts = [0] + np.cumsum(B_sizes).tolist()[:-1]

        # Fill upper diagonal blocks
        for i in range(self.n_modes):
            start = B_starts[i]
            end = start + B_sizes[i]
            xtalk_mask[start:end, start:end] = 1.0

        G_sizes = B_sizes[: self.n_modes - 2]
        if G_sizes:
            G_starts = [0] + np.cumsum(G_sizes).tolist()[:-1]

        row_idx = M + self.n_modes - 1

        # Calculate dependencies for the lower rows (Phis)
        for E in range(1, self.n_modes - 1):
            for S in range(1, E + 1, 2):
                count = (self.n_modes // 2 - E // 2) if S == 1 else 1

                bl_col_start = B_starts[S]
                bl_col_end = B_starts[E] + B_sizes[E]

                g_start_idx = max(0, S - 2)
                g_end_idx = E - 1
                br_col_start = M + self.n_modes - 1 + G_starts[g_start_idx]
                br_col_end = M + self.n_modes - 1 + G_starts[g_end_idx] + G_sizes[g_end_idx]

                for _ in range(count):
                    xtalk_mask[row_idx, bl_col_start:bl_col_end] = 1.0
                    xtalk_mask[row_idx, br_col_start:br_col_end] = 1.0
                    row_idx += 1

        return static_phases_mask, xtalk_mask

    def _generate_mixing_layers(self) -> tuple[int, nparray]:
        """
        Generate the static mixing layers (directional couplers) for the Clements
        architecture. Phase shift layer are excluded.

        Returns
        -------
        tuple
            A tuple containing the number of mixing layers (int) and
            the 3D array of matrices M (np.ndarray).
        """
        n_mixing_layers = 2 * self.n_modes  # Number of mixing layers (EXCLUDING the phase blocks)
        refl = 0.5
        trasm = 1.0 - refl
        BS = torch.tensor([[np.sqrt(trasm), 1j * np.sqrt(refl)], [1j * np.sqrt(refl), np.sqrt(trasm)]])

        # First kind of beam-splitter layer (V1)
        layer_mat = [BS] * (self.n_modes // 2)
        V1 = sp.linalg.block_diag(*layer_mat)

        # Second kind of beam-splitter layer (V2)
        layer_mat = [np.eye(1)] + [BS] * (self.n_modes // 2 - 1) + [np.eye(1)]
        V2 = sp.linalg.block_diag(*layer_mat)

        # Array of matrices that describe the processor (mixing layers)
        M = np.zeros((n_mixing_layers, self.n_modes, self.n_modes), dtype=np.complex128)

        # Populate M with the pattern: V1, V1, V2, V2 repeated
        pattern = itertools.islice(itertools.cycle([V1, V1, V2, V2]), n_mixing_layers)
        for ii, V in enumerate(pattern):
            M[ii] = V

        return n_mixing_layers, M

    def U_from_phases(self, phases: torch.Tensor) -> torch.Tensor:
        """
        Calculate the 2D unitary matrix representation of the processor on the
        basis of the provided phase array.

        Phases are split into internal (0 to n_phases/2 - 1) and
        external (n_phases/2 to n_phases). They are applied alternately across layers:
            - Layer type 1 (V1): n_modes/2 external/internal phases (even positions, starting at mode 0)
            - Layer type 2 (V2): n_modes/2 - 1 external/internal phases (odd positions, starting at mode 1)

        Parameters
        ----------
        phases : torch.Tensor
            Tensor of optical phases in rad. Shape (n_phases,) or (B, n_phases).

        Returns
        -------
        torch.Tensor
            The complex unitary matrix resulting from the propagation.
            Shape (n_modes, n_modes) or (B, n_modes, n_modes).

        Raises
        ------
        ValueError
            If the last dimension of the input array does not match the
            expected number of phases for the mesh.
        """
        # Check dimension to guarantee vectorial shape
        is_1d = phases.ndim == 1
        if is_1d:
            phases = phases.unsqueeze(0)

        # Input validation
        if phases.shape[-1] != self.n_phases:
            raise ValueError(f"Dimension mismatch: expected {self.n_phases} elements, found {phases.shape[-1]}.")

        B = phases.shape[0]
        device = phases.device

        # (B, N, N) identity array
        U = torch.eye(self.n_modes, dtype=torch.complex128, device=device).unsqueeze(0).repeat(B, 1, 1)

        # Pointers for internal and external phases
        int_ptr = 0
        ext_ptr = self.n_phases // 2

        # Sizes for each layer type
        size_v1 = self.n_modes // 2
        size_v2 = self.n_modes // 2 - 1

        idx_v1 = torch.arange(0, self.n_modes, 2, device=device)
        idx_v2 = torch.arange(1, self.n_modes - 1, 2, device=device)

        # Layer structure: (is_internal, size_v, indices)
        layer_specs = [
            (False, size_v1, idx_v1),  # pattern 0: V1 ext
            (True, size_v1, idx_v1),  # pattern 1: V1 int
            (False, size_v2, idx_v2),  # pattern 2: V2 ext
            (True, size_v2, idx_v2),  # pattern 3: V2 int
        ]

        def consume_phases(phase_tensor, ptr, size):
            return phase_tensor[:, ptr : ptr + size], ptr + size

        for ii in range(self.n_mixing_layers):
            is_internal, size_v, indices = layer_specs[ii % 4]

            if is_internal:
                layer_phases, int_ptr = consume_phases(phases, int_ptr, size_v)
            else:
                layer_phases, ext_ptr = consume_phases(phases, ext_ptr, size_v)

            phase_multiplier = torch.ones((B, self.n_modes), dtype=torch.complex128, device=device)
            phase_multiplier[:, indices] = torch.exp(1j * layer_phases.to(torch.complex128))

            # Broadcasting phase multiplier along rows: (B, N, 1)
            U = U * phase_multiplier.unsqueeze(2)

            # Multiplication by the current mixing layer
            U = torch.matmul(self.M[ii], U)

        return U.squeeze(0) if is_1d else U

    def phases_from_U(self, U_target: torch.Tensor, target_infidelity: float = 1e-6, **kwargs) -> torch.Tensor:
        """
        Decomposes a target unitary matrix into the phases of the Clements mesh.

        Parameters
        ----------
        U_target : torch.Tensor
            Target unitary matrix. Shape (n_modes, n_modes) or (B, n_modes, n_modes).
        target_infidelity : float, optional
            Target infidelity threshold. Default is 1e-6.

        Returns
        -------
        torch.Tensor
            Tensor of phases matching the mesh configuration.
        """
        _, Tlist = self.clements_decompose(U=U_target, tol=target_infidelity)

        th = torch.stack([t.theta for t in Tlist], dim=1)
        ph = torch.stack([t.phi for t in Tlist], dim=1)
        phases = torch.cat([th, ph], dim=1)

        # Remapping the redundant external phases (exluding the first n / 2, not involved in any MZI ring)
        idx_start = self.n_phases // 2 + n // 2
        idx_end = self.n_phases // 2 + (n - 1)
        delta = int(idx_end - idx_start)
        counter = 1

        for ii in range(idx_start, idx_end):
            for k in range(counter):
                phases[:, ii + (k + 1) * delta] -= phases[:, ii]
            counter += 2
        return phases

    @staticmethod
    def mzi_matrix(
        m: int,
        n: int,
        theta: torch.Tensor,
        phi: torch.Tensor,
        N: int,
    ) -> torch.Tensor:
        """Build a batch (B) of N×N MZI unitary matrices.

        The non-trivial 2×2 block at column indices (m, n) with n = m+1 is:

        .. code-block:: none

            T[m,m] =  i·exp(iθ/2)·sin(θ/2)·exp(iφ)
            T[m,n] =  i·exp(iθ/2)·cos(θ/2)
            T[n,m] =  i·exp(iθ/2)·cos(θ/2)·exp(iφ)
            T[n,n] = -i·exp(iθ/2)·sin(θ/2)

        All other diagonal entries are 1, off-diagonal entries are 0.

        Parameters
        ----------
        m : int
            Index of the first affected mode.
        n : int
            Index of the second affected mode (n = m+1).
        theta : torch.Tensor, shape (B,)
            Internal phase angles.
        phi : torch.Tensor, shape (B,)
            External phase angles.
        N : int
            Matrix size.

        Returns
        -------
        torch.Tensor, shape (B, N, N)
            Batch of MZI unitary matrices.
        """
        B = theta.shape[0]
        T = torch.eye(N, dtype=torch.complex128, device=theta.device).unsqueeze(0).repeat(B, 1, 1)

        j = torch.tensor(1j, device=theta.device)
        g = j * torch.exp(j * theta / 2)
        s = torch.sin(theta / 2)
        c = torch.cos(theta / 2)
        e = torch.exp(j * phi)

        T[:, m, m] = g * s * e
        T[:, m, n] = g * c
        T[:, n, m] = g * c * e
        T[:, n, n] = -g * s
        return T

    @staticmethod
    def mzi_matrix_dag(
        m: int,
        n: int,
        theta: torch.Tensor,
        phi: torch.Tensor,
        N: int,
    ) -> torch.Tensor:
        """
        Compute the conjugate transpose of an MZI matrix batch.

        Parameters
        ----------
        m : int
            Index of the first affected mode.
        n : int
            Index of the second affected mode (n = m+1).
        theta : torch.Tensor, shape (B,)
            Internal phase angles.
        phi : torch.Tensor, shape (B,)
            External phase angles.
        N : int
            Matrix size.

        Returns
        -------
        torch.Tensor, shape (B, N, N)
            Conjugate transpose T(m, n, θ, φ)^†.
        """
        return torch.adjoint(Clements.mzi_matrix(m, n, theta, phi, N))

    @staticmethod
    def _phases_null_right(row: int, col: int, W: torch.Tensor) -> tuple:
        """
        Solve for (theta, phi) such that (W @ T^†)[row, col] = 0.

        T^† acts on column pair (col, col+1). The annihilation condition is:

        .. code-block:: none

            W[row, col] · conj(T[col, col]) + W[row, col+1] · conj(T[col, col+1]) = 0

        Substituting the MZI entries and simplifying gives the ratio:

        .. code-block:: none

            r = -W[row, col+1] / W[row, col]
            theta = 2·arctan|r|,   phi = -angle(r)

        Parameters
        ----------
        row : int
            Row index of the element to annihilate.
        col : int
            Column index of the element to annihilate.
        W : torch.Tensor, shape (B, N, N)
            Current working matrix batch.

        Returns
        -------
        theta : torch.Tensor, shape (B,)
            Internal phase angles in [0, 2π).
        phi : torch.Tensor, shape (B,)
            External phase angles in [0, 2π).
        """
        a = W[:, row, col]  # element to annihilate, shape (B,)
        b = W[:, row, col + 1]  # neighbouring column, shape (B,)
        B = a.shape[0]

        theta = torch.zeros(B, dtype=torch.float64, device=W.device)
        phi = torch.zeros(B, dtype=torch.float64, device=W.device)

        zero_a = torch.isclose(a, torch.tensor(0.0, dtype=a.dtype, device=W.device))  # already zero
        general = ~zero_a & ~torch.isclose(b, torch.tensor(0.0, dtype=b.dtype, device=W.device))

        theta[zero_a] = torch.pi  # T(π,0) swaps and zeros correctly

        if torch.any(general):
            r = -b[general] / a[general]
            theta[general] = 2 * torch.arctan(torch.abs(r))
            phi[general] = (-torch.angle(r)) % (2 * torch.pi)

        return theta % (2 * torch.pi), phi % (2 * torch.pi)

    @staticmethod
    def _phases_null_left(row: int, col: int, W: torch.Tensor) -> tuple:
        """
        Solve for (theta, phi) such that (T @ W)[row+1, col] = 0.

        T acts on row pair (row, row+1). The annihilation condition is:

        .. code-block:: none

            T[row+1, row] * W[row, col] + T[row+1, row+1] * W[row+1, col] = 0

        Substituting the MZI entries and simplifying gives:

        .. code-block:: none

            r = +W[row, col] / W[row+1, col]
            theta = 2·arctan|r|,   phi = -angle(r)

        Parameters
        ----------
        row : int
            Row index such that W[row+1, col] is the element to annihilate.
        col : int
            Column index of the element to annihilate.
        W : torch.Tensor, shape (B, N, N)
            Current working matrix batch.

        Returns
        -------
        theta : torch.Tensor, shape (B,)
            Internal phase angles in [0, 2π).
        phi : torch.Tensor, shape (B,)
            External phase angles in [0, 2π).
        """
        a = W[:, row + 1, col]  # element to annihilate, shape (B,)
        b = W[:, row, col]  # neighbouring row, shape (B,)
        B = a.shape[0]

        theta = torch.zeros(B, dtype=torch.float64, device=W.device)
        phi = torch.zeros(B, dtype=torch.float64, device=W.device)

        zero_a = torch.isclose(a, torch.tensor(0.0, dtype=a.dtype, device=W.device))
        general = ~zero_a & ~torch.isclose(b, torch.tensor(0.0, dtype=b.dtype, device=W.device))

        theta[zero_a] = torch.pi

        if torch.any(general):
            r = b[general] / a[general]
            theta[general] = 2 * torch.arctan(torch.abs(r))
            phi[general] = (-torch.angle(r)) % (2 * torch.pi)

        return theta % (2 * torch.pi), phi % (2 * torch.pi)

    @staticmethod
    def _phi_left_to_right(
        m: int,
        n: int,
        theta: torch.Tensor,
        W: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert a left MZI op into an equivalent right MZI op.

        After undoing the left-application (W ← T^† @ W), the matrix W has a
        non-zero off-diagonal entry W[m, n]. Solves for phi_new such that:

        .. code-block:: none

            (W @ T(m, n, theta, phi_new)^†)[m, n] = 0

        keeping theta unchanged. The condition simplifies to:

        .. code-block:: none

            r = -W[m, m] / W[m, n]
            phi_new = (angle(r) + π) % 2π

        Parameters
        ----------
        m : int
            Index of the first affected mode.
        n : int
            Index of the second affected mode (n = m+1).
        theta : torch.Tensor, shape (B,)
            Internal phase angles (unchanged from the original left op).
        W : torch.Tensor, shape (B, N, N)
            Working matrix batch after undoing the left application.

        Returns
        -------
        torch.Tensor, shape (B,)
            New external phase angles phi_new for the equivalent right op.
        """
        B = W.shape[0]
        phi_new = torch.zeros(B, dtype=torch.float64, device=W.device)
        w_mm = W[:, m, m]
        w_mn = W[:, m, n]

        general = ~torch.isclose(torch.abs(w_mn), torch.tensor(0.0, dtype=torch.abs(w_mn).dtype, device=W.device))
        if torch.any(general):
            r = -w_mm[general] / w_mn[general]
            phi_new[general] = (torch.angle(r) + torch.pi) % (2 * torch.pi)

        return phi_new

    @staticmethod
    def rearrange_clements(ops: list[MZIOp]) -> list[MZIOp]:
        """
        Reorder MZI ops to match the physical column layout of a Clements mesh.

        In a Clements interferometer, MZIs acting on non-overlapping mode pairs
        can run simultaneously in the same physical column. This function groups
        ops into the earliest possible column via greedy layer assignment and
        returns them in column-major order:

        .. code-block:: none

            col 0: m = 0, 2, 4, ...   (even-mode MZIs)
            col 1: m = 1, 3, 5, ...   (odd-mode MZIs)
            col 2: m = 0, 2, 4, ...
            ...

        .. warning::

            The reordering is purely for physical layout reporting. It does
            **not** preserve the matrix-product order, so the returned list
            must **not** be used for reconstruction.

        Parameters
        ----------
        ops : list[MZIOp]
            MZI ops in computation order, as returned by ``clements_decompose``.

        Returns
        -------
        list[MZIOp]
            MZI ops in column-major (physical mesh) order.
        """
        if not ops:
            return ops

        # Greedy layer assignment: place each op in the first column where
        # neither of its two modes is already occupied.
        layers: list[list[int]] = []  # layers[col] = list of op indices
        for i, op in enumerate(ops):
            placed = False
            for layer in layers:
                occupied = {ops[j].m for j in layer} | {ops[j].n for j in layer}
                if op.m not in occupied and op.n not in occupied:
                    layer.append(i)
                    placed = True
                    break
            if not placed:
                layers.append([i])
        return [ops[i] for layer in layers for i in layer]

    def clements_decompose(
        self,
        U: torch.Tensor,
        tol: float = 1e-11,
    ) -> tuple[torch.Tensor, list[MZIOp]]:
        """
        Batched Clements decomposition of a unitary matrix batch.

        Factorises each matrix in the batch as:

        .. code-block:: none

            U = D @ T_1 @ T_2 @ ... @ T_K

        where D is a diagonal unitary and every T_k is a right-multiplication MZI.

        **Pass 1** — standard Clements alternating sweep:

        - Even diagonals: RIGHT ops  ``W ← W @ T^†``  (zero lower columns)
        - Odd  diagonals: LEFT  ops  ``W ← T @ W``    (zero lower rows)

        After this pass W is diagonal.

        **Pass 2** — convert each LEFT op to an equivalent RIGHT op:

        .. code-block:: none

            W       ← T(theta, phi_old)^† @ W        # undo the left application
            phi_new  = solve( (W @ T^†)[m,n] = 0 )   # same theta, new phi
            W       ← W @ T(theta, phi_new)^†

        The converted op is appended to right_ops.

        Parameters
        ----------
        U : torch.Tensor, shape (B, N, N)
            Batch of B unitary N x N matrices to decompose.
        tol : float, optional
            Absolute tolerance for the unitarity check. Default is 1e-11.

        Returns
        -------
        D : torch.Tensor, shape (B, N, N)
            Diagonal unitary batch.
        right_ops : list[MZIOp]
            All MZI ops in application order. Reconstruct with
            ``U = D @ T_K @ ... @ T_1`` (reversed, using ``mzi_matrix`` not dag).

        Raises
        ------
        ValueError
            If one or more matrices in the batch are not unitary.
        """
        B, _, N = U.shape

        # Unitarity check: U @ U^dag ~= I
        UUd = torch.bmm(U, torch.adjoint(U))
        I = torch.eye(N, dtype=torch.complex128, device=U.device).unsqueeze(0).repeat(B, 1, 1)

        if not torch.allclose(UUd, I, atol=tol, rtol=0):
            raise ValueError("One or more matrices in the batch are not unitary.")

        W: torch.Tensor = U.clone().detach()
        right_ops: list[MZIOp] = []  # ops applied as  W <- W @ T^dag
        left_ops: list[MZIOp] = []  # ops applied as  W <- T  @ W

        for k, i in enumerate(range(N - 2, -1, -1)):
            if k % 2 == 0:
                # RIGHT ops: zero W[row, col] via  W <- W @ T(col, col+1)^dag
                for j in reversed(range(N - 1 - i)):
                    row, col = i + j + 1, j
                    theta, phi = self._phases_null_right(row, col, W)
                    right_ops.append(MZIOp(m=col, n=col + 1, theta=theta, phi=phi))
                    W = torch.bmm(W, self.mzi_matrix_dag(col, col + 1, theta, phi, N))
            else:
                # LEFT ops: zero W[row+1, col] via  W <- T(row, row+1) @ W
                for j in range(N - 1 - i):
                    row, col = i + j, j
                    theta, phi = self._phases_null_left(row, col, W)
                    left_ops.append(MZIOp(m=row, n=row + 1, theta=theta, phi=phi))
                    W = torch.bmm(self.mzi_matrix(row, row + 1, theta, phi, N), W)

        # W is diagonal
        # Convert each LEFT op into an equivalent RIGHT op (batch).
        for op in reversed(left_ops):
            # Undo the left application
            W = torch.bmm(self.mzi_matrix_dag(op.m, op.n, op.theta, op.phi, N), W)
            # Compute phi_new (batch) that re-zeroes W[m, n] via a right op
            phi_new = self._phi_left_to_right(op.m, op.n, op.theta, W)
            # Apply the equivalent right op
            W = torch.bmm(W, self.mzi_matrix_dag(op.m, op.n, op.theta, phi_new, N))

            right_ops.append(MZIOp(m=op.m, n=op.n, theta=op.theta, phi=phi_new))

        # W is diagonal; right_ops contains all transformations.
        return W, self.rearrange_clements(right_ops)


class Bell(Mesh):
    """
    A photonic processor built according to the Bell scheme.

    The number of modes can be specified (it must be EVEN in this version).
    Multiple configurations of phases can be calculated at the same time.
    Thermo-optic phase actuation is also considered in the terms of
    squared currents applied to phase-shifters; phase-shifters are present for
    a subset of the available phase terms.

    Parameters
    ----------
    n_modes : int
        Number of modes (must be even).
    calibration_file : str, optional
        Path to a pickle file containing the hardware calibration data.
    """

    def _generate_masks(self) -> tuple[nparray, nparray]:
        """
        Generate the boolean masks for the Bell architecture, specifying which
        phases and cross-talk elements are physically implemented.

        Returns
        -------
        tuple of np.ndarray
            A tuple containing (static_phases_mask, xtalk_mask).
        """
        # Mask for phases and currents that relate the actual free parameters of calibration to the available ones
        static_phases_mask = np.ones((self.n_modes, self.n_modes - 1))  # n(n-1) overall phases
        static_phases_mask[0, :] = 0  # In the first layer we act only on the even ones
        static_phases_mask[0, ::2] = 1
        static_phases_mask[-1, :] = 0  # In the last layer we act only on the odd ones
        static_phases_mask[-1, 1::2] = 1
        static_phases_mask = static_phases_mask.reshape(-1).astype(bool)  # Reshape into 1D array of bool

        # Mask for cross-talk coefficients
        xtalk_mask = np.ones(
            (self.n_modes, self.n_modes - 1, self.n_modes - 1)
        )  # We set a general framework of n layers with (n-1)x(n-1) coeff.
        xtalk_mask[0, :, :] = 0  # In the first layer we act only on the even modes
        xtalk_mask[-1, :, :] = 0  # In the last layer we act only on the odd ones
        xtalk_mask[0, ::2, ::2] = 1
        xtalk_mask[-1, 1::2, 1::2] = 1
        xtalk_mask = sp.linalg.block_diag(*xtalk_mask)  # Reshape into 2D array of bool

        return static_phases_mask, xtalk_mask

    def _generate_mixing_layers(self) -> tuple[int, np.ndarray]:
        """
        Generate the static mixing layers (directional couplers) for the Bell
        architecture. Phase shift layer are excluded.

        Returns
        -------
        tuple
            A tuple containing the number of mixing layers (int) and
            the 3D array of matrices M (np.ndarray).
        """
        n_mixing_layers = self.n_modes + 1  # Number of mixing layers (EXCLUDING the phase blocks)
        refl = 0.5
        trasm = 1.0 - refl
        BS = np.array([[np.sqrt(trasm), 1j * np.sqrt(refl)], [1j * np.sqrt(refl), np.sqrt(trasm)]])

        # First kind of beam-splitter layer (V1)
        V1 = BS
        for ii in range(1, self.n_modes // 2):
            V1 = sp.linalg.block_diag(V1, BS)

        # Second kind of beam-splitter layer (V2)
        V2 = np.array([[1]])
        for ii in range(1, self.n_modes // 2):
            V2 = sp.linalg.block_diag(V2, BS)
        V2 = sp.linalg.block_diag(V2, np.array([[1]]))

        # Array of matrices that describe the processor (mixing layers)
        M = np.zeros((n_mixing_layers, self.n_modes, self.n_modes), dtype=np.complex128)

        # Populate M
        for ii in range(1, n_mixing_layers, 2):
            M[ii - 1, :, :] = V1 @ V2
            M[ii, :, :] = V2 @ V1
        M[0, :, :] = V1
        M[self.n_modes, :, :] = V2

        return n_mixing_layers, M

    def U_from_phases(self, phases: torch.Tensor) -> torch.Tensor:
        """
        Calculate the 2D unitary matrix representation of the processor on the
        basis of the provided phase array.

        Only n_modes-1 phases per layer are considered; this is sufficiently general
        because it defines the unitary apart from a global phase term. Supports
        batch calculations (B, n_modes, n_modes) simultaneously.

        Parameters
        ----------
        phases : torch.Tensor
            Tensor of optical phases in rad. Shape (n_phases,) or (B, n_phases).

        Returns
        -------
        torch.Tensor
            The complex unitary matrix resulting from the propagation.
            Shape (n_modes, n_modes) or (B, n_modes, n_modes).

        Raises
        ------
        ValueError
            If the last dimension of the input array does not match the
            expected number of phases for the mesh.
        """
        # Check dimension to guarantee vectorial shape
        is_1d = phases.ndim == 1
        if is_1d:
            phases = phases.unsqueeze(0)

        # Input validation
        if phases.shape[-1] != self.n_phases:
            raise ValueError(f"Dimension mismatch: expected {self.n_phases} elements, found {phases.shape[-1]}.")

        B = phases.shape[0]
        device = phases.device

        # (B, N, N) identity array
        U = torch.eye(self.n_modes, dtype=torch.complex128, device=device).unsqueeze(0).repeat(B, 1, 1)

        # Vectorized propagation
        for ii in range(self.n_mixing_layers - 1):
            # Multiplication by the current mixing layer
            U = torch.matmul(self.M[ii], U)

            # Extract phases for the current layer
            # Each layer has exactly (n_modes - 1) phases
            start_idx = ii * (self.n_modes - 1)
            end_idx = (ii + 1) * (self.n_modes - 1)
            layer_phases = phases[:, start_idx:end_idx]

            # Broadcasting: add a column axis to correctly multiply rows
            phase_multiplier = torch.exp(1j * layer_phases.to(torch.complex128)).unsqueeze(2)

            # PYTORCH TRICK: Avoid in-place operations for Autograd compatibility
            # Split and concatenate tensors instead of U[:, 0:N-1, :] = ...
            U_top = U[:, 0 : (self.n_modes - 1), :] * phase_multiplier
            U_bottom = U[:, (self.n_modes - 1) :, :]
            U = torch.cat([U_top, U_bottom], dim=1)

        # Last layer
        U = torch.matmul(self.M[-1], U)

        return U.squeeze(0) if is_1d else U

    def phases_from_U(
        self,
        U_target: torch.Tensor,
        max_iter: int = 100000,
        target_infidelity: float = 1e-6,
        plot_graph: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Given a unitary matrix U_target, finds the phases that implement it
        with the UPP.

        Actually, to find the matrix more phases are used internally.
        Then, phases that are not used in the processor are simplified.
        The final matrix adheres to the target one only up to input/output
        phase terms. Fully supports batch processing for B matrices simultaneously.

        Parameters
        ----------
        U_target : torch.Tensor
            Target unitary matrix. Shape (n_modes, n_modes) or (B, n_modes, n_modes).
        max_iter : int, optional
            Maximum number of iterations.
        target_infidelity : float, optional
            Target infidelity threshold to stop the optimization.
        plot_graph : bool, optional
            If True, plots the infidelity trend over iterations.

        Returns
        -------
        torch.Tensor
            The tensor of optimal phases in rad. Shape (n_phases,) or (B, n_phases).
        """
        # Check dimension to guarantee vectorial shape
        is_2d = U_target.ndim == 2
        if is_2d:
            U_target = U_target.unsqueeze(0)

        # Useful definitions
        B = U_target.shape[0]
        n_matrices = self.n_mixing_layers + 1  # Number of mixing matrices + the inverse of the target
        device = U_target.device

        # Array of mixing matrices with inv_target at the end, over B batches
        v_matrices = torch.zeros((B, n_matrices, self.n_modes, self.n_modes), dtype=torch.complex128, device=device)
        v_matrices[:, :-1] = self.M.unsqueeze(0)
        v_matrices[:, -1] = torch.adjoint(U_target)

        # Phase terms are introduced before each V matrix in this
        # decomposition; they are here initialized to random values
        rand_angles = torch.rand(B, self.n_modes + 2, self.n_modes, dtype=torch.float64, device=device) * 2 * torch.pi
        phase_terms = torch.exp(1j * rand_angles)

        # Start the iterative decomposition
        print(f"Iterative phases optimization (Batch size: {B})")
        fidelity_monitor = torch.zeros((B, max_iter), device=device)

        import time

        start_time = time.time()

        for step in range(max_iter):
            buffer_matrix = torch.eye(self.n_modes, dtype=torch.complex128, device=device).unsqueeze(0).repeat(B, 1, 1)

            # Calculates the matrix product
            for ii in range(n_matrices):
                # Multiply rows (First phases)
                buffer_matrix = buffer_matrix * phase_terms[:, ii, :].unsqueeze(2)
                # Then mixing matrix
                buffer_matrix = torch.matmul(v_matrices[:, ii, :, :], buffer_matrix)

            # Vectorized fidelity check (Batch trace over inner axes)
            trace_b = torch.einsum('bii->b', buffer_matrix)
            fid = torch.abs(trace_b / self.n_modes) ** 2
            fidelity_monitor[:, step] = fid

            # If it is high enough it stops cycling
            if torch.all((1 - fid) < target_infidelity):
                fidelity_monitor = fidelity_monitor[:, :step]
                break

            if step > 0 and step % 1000 == 0:
                print(f"Iteration {step} - Max Infidelity: {torch.max(1 - fid).item():.2e}")

            # If not, it performs another update cycle of all matrices
            for ii in range(n_matrices):
                # Multiplication along columns to expose phases
                buffer_matrix = buffer_matrix * torch.conj(phase_terms[:, ii, :]).unsqueeze(1)

                # Update phase list
                diagonal = torch.diagonal(buffer_matrix, dim1=1, dim2=2).conj()
                phase_terms[:, ii, :] = diagonal / torch.abs(diagonal)

                # Multiplication along rows to apply new phases
                buffer_matrix = buffer_matrix * phase_terms[:, ii, :].unsqueeze(2)

                # Removes next mixing layer
                buffer_matrix = torch.matmul(v_matrices[:, ii, :, :], buffer_matrix)
                buffer_matrix = torch.matmul(buffer_matrix, torch.adjoint(v_matrices[:, ii, :, :]))

        print(f"--- Elapsed time: {(time.time() - start_time):.2f} seconds.")

        # Vectorized cleanup and restoration
        phase_terms = torch.angle(phase_terms)  # Phases are now real numbers
        phase_terms = phase_terms - phase_terms[:, :, -1].unsqueeze(2)  # Subtract the phase of the last mode
        phase_terms = phase_terms[:, 1:-1, :]  # Remove external phases

        phase_terms[:, 0, ::2] -= phase_terms[:, 0, 1::2]  # First layer only even modes
        phase_terms[:, 0, 1::2] = 0.0

        phase_terms[:, -1, 1 : self.n_modes - 1 : 2] -= phase_terms[
            :, -1, 2 : self.n_modes - 1 : 2
        ]  # Last layer only odd modes
        phase_terms[:, -1, : self.n_modes - 1 : 2] = 0.0

        phase_terms = phase_terms[:, :, :-1]  # Removes last mode
        phases = phase_terms.reshape(B, -1)
        phases = phases % (2 * torch.pi)

        # Check dimension in order to provide always squeezed shape
        if is_2d:
            phases = phases[0]
            fidelity_monitor = fidelity_monitor[0]

        # Output for debug
        if plot_graph:
            fig, ax = plt.subplots()
            ax.semilogy((1 - fidelity_monitor.cpu().numpy()).T)
            ax.set_xlabel("Iteration number")
            ax.set_ylabel("Infidelity")
            ax.set_title("Decomposition convergence diagram")
            ax.grid(True, linestyle=":", alpha=0.6)
            plt.show()

        return phases


if __name__ == "__main__":

    from tools import fidelity, get_haar_unitary, make_real_border
    import traceback

    processor = None

    try:
        n = 24
        batch_size = 1000
        processor = Clements(n_modes=n)

        print(f"\n--- Batch Testing ({batch_size} unitaries simultaneously) ---")

        # 1. Target: Generate a batch of Haar random unitaries (and make them "real-bordered")
        u_target_raw = get_haar_unitary(n, batch_size)
        u_target, _, _ = make_real_border(u_target_raw)

        # 2. Decomposition and Parallel Current Calculation
        start_t = time.time()
        phases_target = processor.phases_from_U(u_target, plot_graph=True)
        currents_target = processor.currents_from_phases(phases_target)
        print(f"-> Decomposition and current calculation completed in {time.time()-start_t:.3f}s")

        # 3. Vectorized Forward Propagation
        phases_exp = processor.phases_from_currents(currents_target)
        u_exp_raw = processor.U_from_phases(phases_exp)

        # 4. Final Real-Bordering
        u_exp, _, _ = make_real_border(u_exp_raw)

        # 5. Fidelity Calculation
        fids = fidelity(u_target, u_exp)

        print("-" * 45)
        print(
            "Final Fidelities:\n"
            f"  shape:  {list(fids.shape)}\n"
            f"  dtype:  {fids.dtype}\n"
            f"  device: {fids.device}\n"
            f"  min/max: {fids.min():.2e} / {fids.max():.2e}"
        )

        if torch.all(fids > 0.9999):
            print("Test PASSED: Batch decomposition and propagation are accurate.")
        else:
            print("Warning: Fidelity is below the expected threshold.")
        print("-" * 45)

    except Exception as e:
        traceback.print_exc()

    finally:
        pass
