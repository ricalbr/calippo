"""
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
import pathlib
import pickle
import time
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import torch


class Mesh(torch.nn.Module, ABC):
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
    nonlinear : bool, optional
        If True, a non-linear relation between phases and currents is applied:
        phases = static_phases + xtalk @ (I^2 * (1 + nl_factor * I^2)).
        Default is False (purely quadratic relation).
    """

    def __init__(
        self,
        n_modes: int,
        calibration_file: str | None = None,
        non_linear: bool = False,
        unbalanced_beam_splitters: bool = False,
    ) -> None:
        super().__init__()

        if n_modes % 2 != 0:
            n_modes -= 1
            print("Modes must be EVEN for this version.\n", f"We shall take {n_modes} modes.")

        self.n_modes = n_modes  # Number of modes
        self.n_phases = (n_modes - 1) * n_modes  # Number of tunable parameters
        self.n_used_phases = (n_modes - 1) ** 2  # Number of in-use tunable parameters
        self.non_linear = non_linear  # Thermal phase shifters non-linear flag
        self.unbalanced_beam_splitters = unbalanced_beam_splitters  # Unbalanced beam splitters flag

        # Register as PyTorch buffers for device compatibility (non-trainable)
        # TODO: make two properties for static_phases_mask adn xt_mask
        sp_mask, xt_mask = self._generate_masks()
        self.register_buffer("static_phases_mask", sp_mask)
        self.register_buffer("xtalk_mask", xt_mask)

        # Populates xtalk, static phases, nl_factor, and beam splitters angle
        if calibration_file is not None:
            sp, xt, nl, sigma = self.import_calibration(calibration_file)
        else:
            sp, xt, nl, sigma = self.generate_random_calibration(xtalk_ampl=1.0, static_phases_ampl=2 * torch.pi)

        # Register as torch.nn.Parameters to enable training
        self.static_phases = torch.nn.Parameter(sp)
        self.xtalk = torch.nn.Parameter(xt)
        self.nl_factor = torch.nn.Parameter(nl)
        self.sigma = torch.nn.Parameter(sigma)

        self.M = self._compute_mixing_layer()

        # Output
        on = lambda flag: "Enabled" if flag else "Disabled"
        rows = [
            ("  Modes", self.n_modes),
            ("  Phases", f"{self.n_phases} tunable, {self.n_used_phases} used"),
            ("  Calibration", "none" or self.calibration_file),
            ("  Unbalanced BS", on(self.unbalanced_beam_splitters)),
            ("  Non-linear", on(self.non_linear)),
            ("  Dtype", self.sigma.dtype),
            ("  Device", self.sigma.device),
        ]
        width = max(len(label) for label, _ in rows)
        print(" PROCESSOR MODEL SUMMARY ".center(40, "-"))
        for label, value in rows:
            print(f"{label:<{width}} : {value}")

    @abstractmethod
    def _generate_masks(self) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    @property
    @abstractmethod
    def n_mixing_layers(self) -> int:
        pass

    @abstractmethod
    def _compute_mixing_layer(self) -> torch.Tensor:
        pass

    def import_calibration(self, file_name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Import calibration data from a pickle file.

        Parameters
        ----------
        file_name : str
            Path to the pickle file of the calibration.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            - static_phases (rad)
            - xtalk matrix (rad/mA2)
            - non_linear factor (rad/mA4)
            - MZI splitting ratio angle (rad)

        Raises
        ------
        ValueError
            If the number of modes or phases in the calibration file
            does not match the processor architecture.
        """

        # Load calibration file
        try:
            with open(file_name, "rb") as f:
                calibration = pickle.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Calibration file not found: {file_name}") from e
        except pickle.UnpicklingError as e:
            raise ValueError(f"Failed to load pickle file {file_name}: {e}") from e

        # Input validation
        self._validate_calibration(calibration=calibration)

        # Get static phases vector
        static_phases = torch.as_tensor(calibration["static_phases"], dtype=torch.float64)
        static_phases = torch.remainder(static_phases, 2 * torch.pi)
        static_phases *= self.static_phases_mask.to(torch.float64)  # type: ignore[operator]

        # Get cross-talk matrix
        xtalk = torch.as_tensor(calibration["xt_matrix"], dtype=torch.float64)
        xtalk *= self.xtalk_mask.to(torch.float64)  # type: ignore[operator]

        # Get nl_factor
        if self.non_linear:
            nl_factor = torch.as_tensor(calibration["nl_factor"], dtype=torch.float64)
            nl_factor *= self.static_phases_mask.to(torch.float64)  # type: ignore[operator]
        else:
            nl_factor = torch.zeros(self.n_phases, dtype=torch.float64)

        if self.unbalanced_beam_splitters:
            sigma = torch.as_tensor(calibration["sigma"], dtype=torch.float64)
        else:
            sigma = torch.pi / 4 * torch.ones(self.n_phases // 2, dtype=torch.float64)

        return static_phases, xtalk, nl_factor, sigma

    def _validate_calibration(self, calibration: dict) -> None:
        """Validate calibration dictionary structure and size."""

        required_keys = ["N", "n_heaters", "static_phases", "xt_matrix"]
        missing_keys = [k for k in required_keys if k not in calibration]

        if missing_keys:
            raise KeyError(f"Calibration file missing required keys: {missing_keys}. " f"Expected: {required_keys}")

        if calibration["N"] != self.n_modes:
            raise ValueError(f"Calibration modes mismatch: expected {self.n_modes}, " f"found {calibration['N']}")

        if calibration["n_heaters"] != self.n_phases:
            raise ValueError(
                f"Calibration phases mismatch: expected {self.n_phases}, " f"found {calibration['n_heaters']}"
            )

        if self.non_linear and "nl_factor" not in calibration:
            raise KeyError("Non-linear model is enabled, but 'nl_factor' not found in calibration file.")
        if self.unbalanced_beam_splitters and "sigma" not in calibration:
            raise KeyError("Unbalanced beam splitter model is enabled, but 'sigma' not found in calibration file.")

    def generate_random_calibration(
        self,
        xtalk_ampl: float,
        static_phases_ampl: float,
        noise: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            - static_phases (rad)
            - xtalk matrix (rad/mA2)
            - non_linear factor (rad/mA4)
            - MZI splitting ratio angle (rad)
        """

        # Set noise level
        if noise is None:
            noise = xtalk_ampl / 10.0

        # Build dataset
        sp_mask = torch.as_tensor(self.static_phases_mask, dtype=torch.float64)
        xt_mask = torch.as_tensor(self.xtalk_mask, dtype=torch.float64)

        # Generate static phases
        static_phases = (torch.rand(self.n_phases) * static_phases_ampl) % (2 * torch.pi)
        static_phases = static_phases * sp_mask

        # Generate cross-talk matrix
        xtalk = torch.eye(self.n_phases) * xtalk_ampl
        xtalk += (torch.rand(self.n_phases, self.n_phases) - 0.5) * noise
        xtalk = xtalk * xt_mask

        # Generate non-linear factor
        if self.non_linear:
            nl_factor = torch.full((self.n_phases,), noise / (2 * torch.pi), dtype=torch.float64)
            nl_factor = nl_factor * sp_mask
        else:
            nl_factor = torch.zeros(self.n_phases, dtype=torch.float64)

        # Generate MZI splitting ratio angle
        sigma = torch.pi / 4 * torch.ones(self.n_phases // 2, dtype=torch.float64)

        if self.unbalanced_beam_splitters:
            sigma += 0.0125 * torch.pi * (torch.rand(self.n_phases // 2) - 0.5)  # BS as unbalanced as 45/55
            # sigma += 0.0250 * torch.pi * (torch.rand(self.n_phases // 2) - 0.5)  # BS as unbalanced as 40/60

        return static_phases, xtalk, nl_factor, sigma

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
        coefficients and non-linear factors. Supports batches (B, n_phases)
        or single (n_phases,).

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
        # Input validation
        if currents.shape[-1] != self.n_phases:
            raise ValueError(
                f"Dimension mismatch: expected last dimension to be {self.n_phases}, found {currents.shape[-1]}."
            )

        I2 = currents.to(torch.float64) ** 2
        I2_eff = I2 * (1 + self.nl_factor * I2)

        # Calculate phases (Gradient support): (B, P) @ (P, P).T -> (B, P)
        phases = self.static_phases + I2_eff @ self.xtalk.T
        return torch.remainder(phases, 2 * torch.pi)

    @abstractmethod
    def U_from_phases(self, phases: torch.Tensor) -> torch.Tensor:
        pass

    @staticmethod
    @torch.no_grad()
    def minimize_currents(target_diff, xtalk, max_iter=20, n_bisect=50):
        """
        Bisection on the upper bound `num`: find the smallest num such that the
        solution of xtalk @ I2 = target (with 2*pi phase corrections) lies in [0, num].
        Returns the I2 solution for the best feasible num.
        """
        TWO_PI = torch.tensor(2 * torch.pi, dtype=target_diff.dtype, device=target_diff.device)

        I2_eff = torch.linalg.solve(xtalk, target_diff)

        num_ok = float(I2_eff.mean()) ** 2
        num_ko = float(I2_eff.mean() * 0.1) ** 2

        best_I2 = None
        for _ in range(n_bisect):
            num = (num_ok + num_ko) / 2
            target = target_diff.clone()
            I2 = I2_eff.clone()

            it = 0
            while ((I2 < 0) | (I2 > num)).any() and it < max_iter:
                negative = I2 < 0
                exceeds = I2 > num
                target = target + negative.to(target.dtype) * TWO_PI - exceeds.to(target.dtype) * TWO_PI
                I2 = torch.linalg.solve(xtalk, target)
                it += 1
            feasible = bool(((I2 >= 0) & (I2 <= num)).all())

            if feasible:
                num_ok = num
                best_I2 = I2
            else:
                num_ko = num

        if best_I2 is None:
            raise RuntimeError("No feasible solution found: increase num_ok or max_iter")

        return best_I2

    def currents_from_phases(self, phases: torch.Tensor) -> torch.Tensor:
        """
        Calculate the currents in mA required to achieve target optical phases,
        compensating for cross-talk and non-linear calibration.
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

        # Input validation
        if phases.shape[-1] != self.n_phases:
            raise ValueError(
                f"Dimension mismatch: expected last dimension to be {self.n_phases}, found {phases.shape[-1]}."
            )

        phases = torch.remainder(phases, 2 * torch.pi)

        # Reduce dimensions to used parameters in order to avoid singularities
        phases_used = phases[:, self.static_phases_mask]  # type: ignore[operator]
        static_phases = self.static_phases[self.static_phases_mask].to(torch.float64)  # type: ignore[operator]

        used_rows = torch.any(self.xtalk_mask, dim=0)  # type: ignore[operator]
        xtalk = self.xtalk[used_rows][:, used_rows].to(torch.float64)

        # Numerical inversion: xtalk @ I2_eff = target_diff, with phases wrapped to [0, 2*pi)
        target_diff = torch.remainder((phases_used - static_phases).T, 2 * torch.pi)
        I2_eff = self.minimize_currents(target_diff, xtalk).T

        # Closed-form inversion of the non-linear relation assuming nl_factor > 0
        if self.non_linear:
            nl_factor = self.nl_factor[self.static_phases_mask].to(torch.float64)  # type: ignore[operator]
            I2 = (-1 + torch.sqrt(1 + 4 * nl_factor * I2_eff)) / (2 * nl_factor)
        else:
            I2 = I2_eff

        # Inherit the dtype directly from phases
        I = torch.zeros_like(phases)
        I[:, self.static_phases_mask] = torch.sqrt(I2)  # type: ignore[operator]

        return I

    @abstractmethod
    def phases_from_U(
        self,
        U_target: torch.Tensor,
        target_infidelity: float = 1e-6,
        **kwargs,
    ) -> torch.Tensor:
        pass

    def forward(self, currents):
        # Mask gradients to enforce physical parameters and prevent optimizer artifacts.
        # (the optimizer must not "invent" nonexistent coupling paths or heaters).
        sp_active = self.static_phases * self.static_phases_mask  # type: ignore[operator]
        xt_active = self.xtalk * self.xtalk_mask  # type: ignore[operator]

        if self.non_linear:
            nl_term = self.nl_factor * self.static_phases_mask  # type: ignore[operator]
        else:
            nl_term = torch.zeros_like(self.nl_factor, dtype=torch.float64)

        I2_eff = currents**2 * (1 + nl_term * currents**2)
        phases = sp_active + I2_eff @ xt_active.T

        # update the mixing layer matrix
        if self.unbalanced_beam_splitters:
            self.M = self._compute_mixing_layer()

        U = self.U_from_phases(phases)
        return torch.abs(U)

    def export_calibration(
        self,
        path: str | pathlib.Path,
        state_dict: dict[str, torch.Tensor] | None = None,
    ) -> pathlib.Path:
        """Save the model's static phases, crosstalk matrix and nonlinear factor.

        Parameters
        ----------
        path : str or pathlib.Path
            Output file (parent folders are created).
        state_dict : dict, optional
            Weights to export instead of the model's current ones, e.g. the pipeline's
            ``best_weights``. Must contain ``static_phases``, ``xtalk`` and ``nl_factor``.

        Values are masked the same way ``Mesh.forward`` masks them, so the file holds the
        effective parameters. ``nl_factor`` is always written, which also lets a nonlinear
        model be rebuilt from the file.
        """
        source = state_dict if state_dict is not None else self.state_dict()

        def param(name: str) -> torch.Tensor:
            return source[name].detach().to(dtype=torch.float64)

        static = param("static_phases")
        xtalk = param("xtalk")
        nl_factor = param("nl_factor")

        sp_mask = self.static_phases_mask.detach().to(dtype=torch.float64)
        xt_mask = self.xtalk_mask.detach().to(dtype=torch.float64)

        calibration = {
            "N": int(self.n_modes),
            "n_heaters": int(self.n_phases),
            "static_phases": torch.remainder(static * sp_mask, 2 * torch.pi),
            "xt_matrix": xtalk * xt_mask,
            "nl_factor": nl_factor * sp_mask.to(nl_factor.device),
            "sigma": param("sigma"),
        }

        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(calibration, path)

        print(f"Calibration exported to {path}")
        return path

    def load_calibration(self, path: str | pathlib.Path) -> None:
        """Load a calibration file into an existing model, in place.

        Reuses ``model.import_calibration``, so size checks and masking behave exactly as when
        passing ``calibration_file=`` to the constructor. For a linear model (``nonlinear=False``)
        the file's ``nl_factor`` is ignored and the model's is set to zero.
        """
        static_phases, xtalk, nl_factor, sigma = self.import_calibration(str(path))

        with torch.no_grad():
            for param, value in (
                (self.static_phases, static_phases),
                (self.xtalk, xtalk),
                (self.nl_factor, nl_factor),
                (self.sigma, sigma),
            ):
                param.copy_(torch.from_numpy(np.asarray(value)).to(device=param.device, dtype=param.dtype))

        print(f"Calibration loaded from {path}")


@dataclasses.dataclass
class MZIOp:
    """
    One batched MZI operation

    Attributes
    ----------
    index : index of the MZI in the rectangular mesh (column-wise, MZI0 is the top-left one)
    m, n  : column indices of the 2×2 block  (m = n-1)
    theta : internal (splitting) phase, shape (B,)
    phi   : external phase, shape (B,)
    sigma    : splitting ratio of the beam splitters, shape (B,). Deafult is balanced operation, sigma = pi/4
    """

    index: int
    m: int
    n: int
    theta: torch.Tensor
    phi: torch.Tensor
    sigma: torch.Tensor = torch.tensor(torch.pi / 4)

    def __iter__(self):
        return iter((self.m, self.n, self.theta, self.phi, self.sigma))

    def __repr__(self):
        return f"MZI {self.index}"


class Clements(Mesh):
    """
    A photonic processor built according to the Clements scheme.
    """

    def __init__(
        self,
        n_modes: int,
        calibration_file: str | None = None,
        non_linear: bool = False,
        unbalanced_beam_splitters: bool = False,
    ) -> None:
        super().__init__(
            n_modes=n_modes,
            calibration_file=calibration_file,
            non_linear=non_linear,
            unbalanced_beam_splitters=unbalanced_beam_splitters,
        )

    def _generate_masks(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate the boolean masks for the Clements architecture, specifying
        which phases and cross-talk elements are physically implemented.

        Returns
        -------
        tuple of np.ndarray
            A tuple containing (static_phases_mask, xtalk_mask).
        """

        # Generate static_phases_mask
        static_phases_mask = torch.ones(self.n_phases, dtype=torch.bool)
        half_idx = self.n_phases // 2
        static_phases_mask[half_idx : half_idx + self.n_modes - 1] = False

        # Initialize xtalk_mask
        xtalk_mask = torch.zeros((self.n_phases, self.n_phases), dtype=torch.bool)

        # Calculate main blocks for the first M rows (Thetas)
        mzi = self.n_modes * (self.n_modes - 1) // 2
        B_sizes = torch.tensor([self.n_modes // 2, self.n_modes // 2 - 1] * (self.n_modes // 2))

        # Fill upper diagonal blocks
        start = 0
        for i in range(self.n_modes):
            end = start + B_sizes[i]
            xtalk_mask[start:end, start:end] = True
            start = end

        B_starts = torch.cat([torch.tensor([0]), torch.cumsum(B_sizes[:-1], dim=0)])
        G_sizes = B_sizes[: self.n_modes - 2]
        if len(G_sizes) > 0:
            G_starts = torch.cat([torch.tensor([0]), torch.cumsum(G_sizes[:-1], dim=0)])
        else:
            G_starts = torch.tensor([])

        row_idx = mzi + self.n_modes - 1

        # Calculate dependencies for the lower rows (Phis)
        for E in range(1, self.n_modes - 1):
            for S in range(1, E + 1, 2):
                count = (self.n_modes // 2 - E // 2) if S == 1 else 1

                bl_col_start = B_starts[S].item()
                bl_col_end = B_starts[E].item() + B_sizes[E].item()

                g_start_idx = max(0, S - 2)
                g_end_idx = E - 1
                br_col_start = mzi + self.n_modes - 1 + G_starts[g_start_idx].item()
                br_col_end = mzi + self.n_modes - 1 + G_starts[g_end_idx].item() + G_sizes[g_end_idx].item()

                for _ in range(count):
                    xtalk_mask[row_idx, bl_col_start:bl_col_end] = True
                    xtalk_mask[row_idx, br_col_start:br_col_end] = True
                    row_idx += 1

        return static_phases_mask, xtalk_mask

    def _bs_tensor(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Generate the non-balanced BS matrix modelled with angle σ.

        BS(σ) = [[cos(σ), i*sin(σ)],
                 [i*sin(σ), cos(σ)]]

        Parameters
        ----------
        sigma : float
            BS angular parameter.

        Returns
        -------
        torch.Tensor
            2x2 BS complex matrix (15, 2, 2)
        """

        c = torch.cos(sigma)
        s = torch.sin(sigma)

        BS = torch.stack([torch.stack([c, 1j * s], dim=-1), torch.stack([1j * s, c], dim=-1)], dim=-2).to(
            torch.complex128
        )

        return BS

    @property
    def n_mixing_layers(self) -> int:
        return 2 * self.n_modes  # Number of mixing layers (EXCLUDING the phase blocks)

    def _compute_mixing_layer(self) -> torch.Tensor:
        """
        Generate the static mixing layers (directional couplers) for the Clements
        architecture. Phase shift layer are excluded.

        Returns
        -------
        torch.Tensor
            Tensor of matrices M (2N, N, N).
        """
        BS = self._bs_tensor(self.sigma)

        eye_1 = torch.eye(1, dtype=torch.complex128)
        cycle = [self.n_modes // 2, self.n_modes // 2 - 1]
        idx = 0
        pattern = []

        while idx < self.n_phases // 2:
            repeat_count = cycle[len(pattern) % 2]
            end_idx = min(idx + repeat_count, self.n_phases // 2)

            if len(pattern) % 2 == 0:
                L = torch.block_diag(*BS[idx:end_idx])
            else:
                L = torch.block_diag(eye_1, *BS[idx:end_idx], eye_1)
            pattern.append(L)
            idx = end_idx
        M = torch.stack(pattern)

        # BS(sigma) are assumed identical in a single MZI we repeat the mixing matrices twice
        return M.repeat_interleave(2, dim=0)

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

        # Input validation
        if phases.shape[-1] != self.n_phases:
            raise ValueError(f"Dimension mismatch: expected {self.n_phases} elements, found {phases.shape[-1]}.")

        phases = phases.clone()  # Evita di modificare l'input

        idx_start = self.n_modes // 2
        idx_end = self.n_modes - 1
        offset = self.n_phases // 2
        delta = int(self.n_modes // 2 - 1)
        counter = 1

        # Inverti il remapping: aggiungi invece di sottrarre
        for ii in range(idx_start + offset, idx_end + offset):
            for k in range(counter):
                phases[:, ii + (k + 1) * delta] += phases[:, ii]
            counter += 2

        B = phases.shape[0]

        # (B, N, N) identity array
        U = torch.eye(self.n_modes, dtype=torch.complex128).unsqueeze(0).repeat(B, 1, 1)

        # Pointers for internal and external phases
        int_ptr = 0
        ext_ptr = self.n_phases // 2

        # Sizes for each layer type
        size_v1 = self.n_modes // 2
        size_v2 = self.n_modes // 2 - 1

        idx_v1 = torch.arange(0, self.n_modes, 2)
        idx_v2 = torch.arange(1, self.n_modes - 1, 2)

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

            phase_multiplier = torch.ones((B, self.n_modes), dtype=torch.complex128)
            phase_multiplier[:, indices] = torch.exp(1j * layer_phases).to(torch.complex128)

            # Broadcasting phase multiplier along rows: (B, N, 1)
            U = U * phase_multiplier.unsqueeze(2)

            # Multiplication by the current mixing layer
            U = torch.matmul(self.M[ii], U)
        return U

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
        _, Tlist = self.decompose(U=U_target, tol=target_infidelity)

        th = torch.stack([t.theta for t in Tlist], dim=1)
        ph = torch.stack([t.phi for t in Tlist], dim=1)
        phases = torch.cat([th, ph], dim=1)

        # Remapping the redundant external phases (exluding the first n / 2, not involved in any MZI ring)
        idx_start = self.n_modes // 2
        idx_end = self.n_modes - 1
        offset = self.n_phases // 2
        delta = int(self.n_modes // 2 - 1)
        counter = 1

        for ii in range(idx_start + offset, idx_end + offset):
            for k in range(counter):
                phases[:, ii + (k + 1) * delta] -= phases[:, ii]
            counter += 2
        return torch.remainder(phases, 2 * torch.pi)

    @staticmethod
    def mzi_matrix(
        m: int,
        n: int,
        theta: torch.Tensor,
        phi: torch.Tensor,
        sigma: torch.Tensor,
        N: int,
    ) -> torch.Tensor:
        """Build a batch (B) of N×N MZI unitary matrices.

        The non-trivial 2×2 block at column/row indices (m, n) is defined by:

        T[m,m] = exp(iφ) * (exp(iθ)cos²(σ) - sin²(σ))
        T[m,n] = i * exp(iθ/2) * cos(θ/2) * sin(2σ)
        T[n,m] = i * exp(i(φ + θ/2)) * cos(θ/2) * sin(2σ)
        T[n,n] = cos²(σ) - exp(iθ)sin²(σ)

        All other diagonal entries are 1, off-diagonal entries are 0.

        Parameters
        ----------
        m : int
            Index of the first affected mode.
        n : int
            Index of the second affected mode (typically n = m + 1).
        theta : torch.Tensor, shape (B,)
            Internal phase angles.
        phi : torch.Tensor, shape (B,)
            External phase angles.
        sigma : torch.Tensor
            Beam splitter angle.
        N : int
            Matrix size.

        Returns
        -------
        torch.Tensor, shape (B, N, N)
            Batch of MZI unitary matrices in complex128.

        Sources
        -------
        "Mitigating linear optics imperfections via port allocation and compilation"
        https://arxiv.org/html/2103.03183v1
        """
        B = theta.shape[0]
        T = torch.eye(N, dtype=torch.complex128).unsqueeze(0).repeat(B, 1, 1)

        j = 1j

        T[:, m, m] = torch.exp(j * phi) * (torch.exp(j * theta) * torch.cos(sigma) ** 2 - torch.sin(sigma) ** 2)
        T[:, m, n] = j * torch.exp(j * theta / 2) * torch.cos(theta / 2) * torch.sin(2 * sigma)
        T[:, n, m] = j * torch.exp(j * (phi + theta / 2)) * torch.cos(theta / 2) * torch.sin(2 * sigma)
        T[:, n, n] = -torch.exp(j * theta) * torch.sin(sigma) ** 2 + torch.cos(sigma) ** 2
        return T.to(torch.complex128)

    @staticmethod
    def mzi_matrix_dag(
        m: int,
        n: int,
        theta: torch.Tensor,
        phi: torch.Tensor,
        sigma: torch.Tensor,
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
        sigma : torch.Tensor, optional
            Beam splitter angle.
        N : int
            Matrix size.

        Returns
        -------
        torch.Tensor, shape (B, N, N)
            Conjugate transpose T(m, n, θ, φ, σ)^†.
        """
        return torch.adjoint(Clements.mzi_matrix(m, n, theta, phi, sigma, N))

    @staticmethod
    def _phases_null_right(row: int, col: int, W: torch.Tensor, sigma: torch.Tensor) -> tuple:
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

        s = torch.cos(torch.atan(torch.abs(b / a))) / torch.sin(2 * sigma)
        s = s.clamp(-1, 1)
        theta = 2 * torch.acos(s)

        phi = (
            -torch.angle(b / a)
            - torch.pi / 2
            + theta / 2
            + torch.atan2(
                -torch.cos(sigma) ** 2 * torch.sin(theta),
                (-torch.sin(sigma) ** 2 + torch.cos(sigma) ** 2 * torch.cos(theta)),
            )
        )

        return torch.remainder(theta, 2 * torch.pi), torch.remainder(phi, 2 * torch.pi)

    @staticmethod
    def _phases_null_left(row: int, col: int, W: torch.Tensor, sigma: torch.Tensor) -> tuple:
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

        s = torch.cos(torch.atan(torch.abs(b / a))) / torch.sin(2 * sigma)
        s = s.clamp(-1, 1)
        theta = 2 * torch.acos(s)

        phi = (
            -torch.angle(b / a)
            + torch.pi / 2
            - theta / 2
            + torch.atan2(
                -torch.sin(sigma) ** 2 * torch.sin(theta),
                (torch.cos(sigma) ** 2 - torch.sin(sigma) ** 2 * torch.cos(theta)),
            )
        )

        return torch.remainder(theta, 2 * torch.pi), torch.remainder(phi, 2 * torch.pi)

    @staticmethod
    def _push_left_to_right(
        m: int,
        n: int,
        W: torch.Tensor,
        theta: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert a left MZI op into an equivalent right MZI op.

        After undoing the left-application (W ← T^† @ W), the matrix W has a
        non-zero off-diagonal entry W[m, n]. Solves for phi_new such that:

        .. code-block:: none

            (W @ T(m, n, theta, phi_new)^†)[m, n] = 0

        keeping theta unchanged. The condition simplifies to:

        .. code-block:: none

            r = W[m, m] / W[m, n]
            phi_new = angle(r) % 2π

        Parameters
        ----------
        m : int
            Index of the first affected mode.
        n : int
            Index of the second affected mode (n = m+1).
        W : torch.Tensor, shape (B, N, N)
            Working matrix batch after undoing the left application.

        Returns
        -------
        torch.Tensor, shape (B,)
            New external phase angles phi_new for the equivalent right op.
        """

        phi_new = (
            -torch.angle(W[:, m, n] / W[:, m, m])
            + torch.pi / 2
            - theta / 2
            + torch.atan2(
                -torch.sin(sigma) ** 2 * torch.sin(theta),
                (torch.cos(sigma) ** 2 - torch.sin(sigma) ** 2 * torch.cos(theta)),
            )
        )

        return torch.remainder(phi_new, 2 * torch.pi)

    def decompose(
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
        I = torch.eye(N, dtype=torch.complex128).unsqueeze(0).repeat(B, 1, 1)

        if not torch.allclose(UUd, I, atol=tol, rtol=0):
            raise ValueError("One or more matrices in the batch are not unitary.")

        W: torch.Tensor = U.clone().detach()
        right_ops: list[MZIOp] = []  # ops applied as  W <- W @ T^dag
        left_ops: list[MZIOp] = []  # ops applied as  W <- T  @ W

        # counter to select MZI from the mesh during decomposition (alternating front and bottom)
        start_idx = 0
        end_idx = self.n_phases // 2 - 1

        global_counter = 0
        for k, i in enumerate(range(N - 2, -1, -1)):
            if k % 2 == 0:  # RIGHT ops: zero W[row, col] via  W <- W @ T(col, col+1)^dag
                for counter, j in enumerate(reversed(range(N - 1 - i))):
                    row, col = i + j + 1, j
                    idx = start_idx + (self.n_modes // 2 - 1) * counter
                    sigma = self.sigma[idx]
                    # print(f"R index: {idx}")
                    theta, phi = self._phases_null_right(row, col, W, sigma)
                    right_ops.append(MZIOp(m=col, n=col + 1, theta=theta, phi=phi, sigma=sigma, index=idx))

                    # old_val = W[:, row, col].item()
                    W = torch.bmm(W, self.mzi_matrix_dag(m=col, n=col + 1, theta=theta, phi=phi, sigma=sigma, N=N))
                    global_counter += 1

                start_idx += 1

            else:  # LEFT ops: zero W[row+1, col] via  W <- T(row, row+1) @ W
                for counter, j in enumerate(range(N - 1 - i)):
                    row, col = i + j, j
                    idx = end_idx - (self.n_modes // 2 - 1) * counter
                    sigma = self.sigma[idx]
                    # print(f"L index: {idx}")
                    theta, phi = self._phases_null_left(row, col, W, sigma)
                    left_ops.append(MZIOp(m=row, n=row + 1, theta=theta, phi=phi, sigma=sigma, index=idx))

                    # old_val = W[:, row + 1, col].item()
                    W = torch.bmm(self.mzi_matrix(m=row, n=row + 1, theta=theta, phi=phi, sigma=sigma, N=N), W)
                    global_counter += 1

                end_idx -= 1

        # Check if W is diagonal
        diff = torch.abs(W - torch.diag_embed(torch.diagonal(W, dim1=-2, dim2=-1)))
        is_non_diagonal = (diff > 1e-12).any(dim=(-2, -1))
        non_diag_indices = torch.where(is_non_diagonal)[0]

        if non_diag_indices.numel() > 0:
            print(f"  Found {len(non_diag_indices)} non-diagonal matrices at indices")
        else:
            print("  All matrices are diagonal")

        # W is diagonal, convert each LEFT op into an equivalent RIGHT op (batch).
        for op in reversed(left_ops):
            # Undo the left application
            W = torch.bmm(self.mzi_matrix_dag(m=op.m, n=op.n, theta=op.theta, phi=op.phi, sigma=op.sigma, N=N), W)

            # Compute phi_new that re-zeroes W[m, n] via a right op
            phi_new = self._push_left_to_right(m=op.m, n=op.n, W=W, theta=op.theta, sigma=op.sigma)

            # Apply the equivalent right op
            W = torch.bmm(W, self.mzi_matrix_dag(m=op.m, n=op.n, theta=op.theta, phi=phi_new, sigma=op.sigma, N=N))

            right_ops.append(MZIOp(m=op.m, n=op.n, theta=op.theta, phi=phi_new, sigma=op.sigma, index=op.index))

        return W, sorted(right_ops, key=lambda x: x.index)


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

    def __init__(
        self,
        n_modes: int,
        calibration_file: str | None = None,
        non_linear: bool = False,
        unbalanced_beam_splitters: bool = False,
    ) -> None:
        super().__init__(
            n_modes=n_modes,
            calibration_file=calibration_file,
            non_linear=non_linear,
            unbalanced_beam_splitters=unbalanced_beam_splitters,
        )

    def _generate_masks(self) -> tuple[torch.Tensor, torch.Tensor]:
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

    @property
    def n_mixing_layers(self) -> int:
        return self.n_modes + 1  # Number of mixing layers (EXCLUDING the phase blocks)

    def _compute_mixing_layer(self) -> torch.Tensor:
        """
        Generate the static mixing layers (directional couplers) for the Bell
        architecture. Phase shift layer are excluded.

        Returns
        -------
        tuple
            A tuple containing the number of mixing layers (int) and
            the 3D array of matrices M (np.ndarray).
        """
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
        M = np.zeros((self.n_mixing_layers, self.n_modes, self.n_modes), dtype=np.complex128)

        # Populate M
        for ii in range(1, self.n_mixing_layers, 2):
            M[ii - 1, :, :] = V1 @ V2
            M[ii, :, :] = V2 @ V1
        M[0, :, :] = V1
        M[self.n_modes, :, :] = V2

        return M

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

        # (B, N, N) identity array
        U = torch.eye(self.n_modes, dtype=torch.complex128).unsqueeze(0).repeat(B, 1, 1)

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

        # Array of mixing matrices with inv_target at the end, over B batches
        v_matrices = torch.zeros((B, n_matrices, self.n_modes, self.n_modes), dtype=torch.complex128)
        v_matrices[:, :-1] = self.M.unsqueeze(0)
        v_matrices[:, -1] = torch.adjoint(U_target)

        # Phase terms are introduced before each V matrix in this
        # decomposition; they are here initialized to random values
        rand_angles = torch.rand(B, self.n_modes + 2, self.n_modes, dtype=torch.float64) * 2 * torch.pi
        phase_terms = torch.exp(1j * rand_angles)

        # Start the iterative decomposition
        print(f"Iterative phases optimization (Batch size: {B})")
        fidelity_monitor = torch.zeros((B, max_iter))

        for step in range(max_iter):
            buffer_matrix = torch.eye(self.n_modes, dtype=torch.complex128).unsqueeze(0).repeat(B, 1, 1)

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
            ax.semilogy((1 - fidelity_monitor.detach().numpy()).T)
            ax.set_xlabel("Iteration number")
            ax.set_ylabel("Infidelity")
            ax.set_title("Decomposition convergence diagram")
            ax.grid(True, linestyle=":", alpha=0.6)
            plt.show()

        return phases


if __name__ == "__main__":

    from calippo.tools import fidelity, get_haar_unitary, make_real_border
    import time
    import traceback

    processor = None

    try:
        n = 24
        batch_size = 100
        plot = True

        # Initialize the processor activating the non-linear model
        processor = Clements(n_modes=n, non_linear=True, unbalanced_beam_splitters=True)

        print(" Decomposition ".center(55, '-'))

        # Generate a batch of Haar random unitaries (and make them "real-bordered")
        u_target_raw = get_haar_unitary(n, batch_size)
        u_target, _, _ = make_real_border(u_target_raw)

        # Decomposition and Current-from-Phases Calculation
        start_t = time.perf_counter()
        phases_target = processor.phases_from_U(u_target, plot_graph=True)
        currents_target = processor.currents_from_phases(phases_target)
        print(f"  Batch size: {batch_size}\n" f"  Matrix-to-currents in {time.perf_counter()-start_t:.3f} s")

        # Experimental matrix retrieval
        u_exp_raw = processor.U_from_phases(phases_target)
        u_exp, _, _ = make_real_border(u_exp_raw)

        # Fidelity Calculation
        fids = fidelity(u_target, u_exp)

        print(" Fidelity calculation ".center(55, '-'))
        print(
            f"  mean:    {fids.mean():.10f}\n"
            f"  min/max: {fids.min():.4f} / {fids.max():.4f}\n"
            f"  shape:   {list(fids.shape)}\n"
            f"  dtype:   {fids.dtype}\n"
        )
        print('-' * 55)

        if plot:
            plt.figure()
            plt.hist(fids.tolist(), bins=50, alpha=0.5, edgecolor='k')
            plt.title("Fidelity distribution")
            plt.xlabel("Fidelity [a.u.]")
            plt.ylabel("Frequency")
            plt.show()

        if torch.all(fids > 0.999):
            print("Test PASSED:\nDecomposition and propagation are accurate.")
        else:
            print("Warning: Fidelity is below the expected threshold.")
        print("-" * 55)

    except Exception as e:
        traceback.print_exc()
