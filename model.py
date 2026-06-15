#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 16:07:49 2026

@author: Andrea Crespi, Francesco Ceccarelli

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

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time
import pickle
from abc import ABC, abstractmethod
from typing import Tuple, Optional


###############################################################################
#                           BASE ABSTRACT CLASS                               #
###############################################################################

class Mesh(ABC):
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
    def __init__(self, n_modes: int, calibration_file: Optional[str] = None) -> None:

        if n_modes % 2 != 0:
            n_modes -= 1
            print("Modes must be EVEN for this version.")
            print(f"We shall take {n_modes} modes.")
            
        self.n_modes = n_modes                                      # Number of modes
        self.n_phases = (n_modes - 1) * n_modes                     # Number of tunable parameters
        self.n_used_phases = (n_modes - 1) * (n_modes - 1)          # Number of tunable parameters actually used in the experiments
                        
        # This part builds the specific topology of the mesh (implemented by children)
        self.static_phases_mask, self.xtalk_mask = self._generate_masks()    # Masks that relate the used calibration parameters to the available ones 
        self.n_mixing_layers, self.M = self._generate_mixing_layers()        # Mode mixing implemented by directional couplers layers

        # This part populates xtalk (rad/mA2) and static phases (rad)
        if calibration_file is not None:
            self.static_phases, self.xtalk = self.import_calibration(calibration_file)
        else:
            self.static_phases, self.xtalk = self.generate_random_calibration(1.0, 2 * np.pi) 

        # Output
        print("A new processor was created.")
        print(f"This has {self.n_modes} modes, {self.n_phases} tunable phases.")
        print(f"Only {self.n_used_phases} phases are actually used.")
        print()
    
    
    @abstractmethod
    def _generate_masks(self) -> Tuple[np.ndarray, np.ndarray]:
        pass


    @abstractmethod
    def _generate_mixing_layers(self) -> Tuple[int, np.ndarray]:
        pass


    def import_calibration(self, fileName: str) -> Tuple[np.ndarray, np.ndarray]:
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
        with open(fileName, 'rb') as f:
            calibration = pickle.load(f)

        # Input validation
        if calibration['N'] != self.n_modes:
            raise ValueError(f"Calibration size mismatch: expected {self.n_modes} modes, "
                             f"found {calibration['N']}.")
            
        if calibration['n_heaters'] != self.n_phases:
            raise ValueError(f"Calibration size mismatch: expected {self.n_phases} phases, "
                             f"found {calibration['n_heaters']}.")
        
        # Fetch dataset
        static_phases = np.mod(calibration['static_phases'], 2 * np.pi)
        static_phases = static_phases * self.static_phases_mask
        xtalk = calibration['xt_matrix']
        xtalk = xtalk * self.xtalk_mask

        return static_phases, xtalk
        

    def generate_random_calibration(self, xtalk_ampl: float, static_phases_ampl: float, noise: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
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
        
        # Build dataset
        static_phases = np.mod(np.random.rand(self.n_phases) * static_phases_ampl, 2 * np.pi)
        static_phases = static_phases * self.static_phases_mask
        xtalk = np.eye(self.n_phases) * xtalk_ampl + (np.random.rand(self.n_phases, self.n_phases) - 0.5) * noise
        xtalk = xtalk * self.xtalk_mask

        return static_phases, xtalk


    def generate_random_currents(self, current_ampl: float, batch_size: Optional[int] = None) -> np.ndarray:
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
        np.ndarray
            Array of randomized currents in mA.
        """
        # Check dimension in order to provide always squeezed shape
        if batch_size is None:
            I = np.sqrt(np.random.rand(self.n_phases)) * current_ampl
        else:
            I = np.sqrt(np.random.rand(batch_size, self.n_phases)) * current_ampl
          
        # Return only useful currents
        I *= self.static_phases_mask
        
        return I


    def phases_from_currents(self, currents: np.ndarray) -> np.ndarray:
        """
        Calculate the optical phases for currents based on the cross-talk 
        coefficients. Supports batches (B, n_phases) or single (n_phases,).

        Parameters
        ----------
        currents : np.ndarray
            Array of currents in mA.

        Returns
        -------
        np.ndarray
            Array of calculated phases in rad.

        Raises
        ------
        ValueError
            If the last dimension of the input array does not match the 
            expected number of phases for the mesh.
        """
        # Check dimension to guarantee vectorial shape
        is_1d = currents.ndim == 1
        if is_1d:
            currents = currents[np.newaxis, :]

        # Input validation
        if currents.shape[-1] != self.n_phases:
            raise ValueError(f"Dimension mismatch: expected last dimension to be {self.n_phases}, found {currents.shape[-1]}.")
            
        # Calculate phases: (B, P) @ (P, P).T -> (B, P)
        phases = self.static_phases + (currents ** 2) @ self.xtalk.T
        phases = np.mod(phases, 2 * np.pi)
        
        return phases[0] if is_1d else phases
    

    @abstractmethod
    def U_from_phases(self, phases: np.ndarray) -> np.ndarray:
        pass
        
    
    def currents_from_phases(self, phases: np.ndarray) -> np.ndarray:
        """
        Calculate the currents in mA required to achieve target optical phases,
        compensating for cross-talk using the calibration matrix. 
        Supports single or batch processing.

        Parameters
        ----------
        phases : np.ndarray
            Array of target phase values in rad. Shape (n_phases,) or (B, n_phases).

        Returns
        -------
        np.ndarray
            Array of currents in mA. Shape (n_phases,) or (B, n_phases).

        Raises
        ------
        ValueError
            If the last dimension of the input array does not match the 
            expected number of phases for the mesh.
        """
        # Check dimension to guarantee vectorial shape
        is_1d = phases.ndim == 1
        if is_1d:
            phases = phases[np.newaxis, :]

        # Input validation
        if phases.shape[-1] != self.n_phases:
            raise ValueError(f"Dimension mismatch: expected last dimension to be {self.n_phases}, found {phases.shape[-1]}.")
        phases = np.mod(phases, 2 * np.pi)

        # Reduce dimensions to used parameters in order to avoid singularities
        phases_used = phases[:, self.static_phases_mask]
        static_phases = self.static_phases[self.static_phases_mask]
        used_rows = np.any(self.xtalk_mask, axis=0)
        xtalk = self.xtalk[np.ix_(used_rows, used_rows)]

        # Numerical inversion.
        # The equation is: xtalk @ I2^T = (phases_used - static_phases)^T
        # We transpose to align the batch axis (B) as the last dimension
        target_diff = (phases_used - static_phases).T
        I2 = np.linalg.solve(xtalk, target_diff)
        while np.any(I2 < 0):
            target_diff[I2 < 0] += 2 * np.pi
            I2 = np.linalg.solve(xtalk, target_diff)
        
        # Transpose back and reconstruct original dimensions
        I2 = I2.T
        I = np.zeros_like(phases, dtype=float)
        I[:, self.static_phases_mask] = np.sqrt(I2)
        
        return I[0] if is_1d else I


    @abstractmethod
    def phases_from_U(self, U_target: np.ndarray, max_iter: int = 100000, target_infidelity: float = 1e-6, plot_graph: bool = False) -> np.ndarray:
        pass


###############################################################################
#                               CLASS CLEMENTS                                #
###############################################################################
 
class Clements(Mesh):
    """
    A photonic processor built according to the Clements scheme.
    """
    def _generate_masks(self) -> Tuple[np.ndarray, np.ndarray]:
        pass
    

    def _generate_mixing_layers(self) -> Tuple[int, np.ndarray]:
        pass
        
    
    def U_from_phases(self, phases: np.ndarray) -> np.ndarray:
        pass


    def phases_from_U(self, U_target: np.ndarray, max_iter: int = 100000, target_infidelity: float = 1e-6, plot_graph: bool = False) -> np.ndarray:        
        pass


###############################################################################
#                               CLASS BELL                                    #
###############################################################################
 
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
    def _generate_masks(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate the boolean masks for the Bell architecture, specifying which 
        phases and cross-talk elements are physically implemented.

        Returns
        -------
        tuple of np.ndarray
            A tuple containing (static_phases_mask, xtalk_mask).
        """
        # Mask for phases and currents that relate the actual free parameters of calibration to the available ones       
        static_phases_mask = np.ones((self.n_modes, self.n_modes - 1)) # We have n(n-1) overall phases
        static_phases_mask[0, :] = 0        # In the first layer we act only on the even ones
        static_phases_mask[0, ::2] = 1
        static_phases_mask[-1, :] = 0       # In the last layer we act only on the odd ones
        static_phases_mask[-1, 1::2] = 1
        static_phases_mask = static_phases_mask.reshape(-1).astype('bool') # Reshape into 1D array of bool
        
        # Mask for cross-talk coefficients
        xtalk_mask = np.ones((self.n_modes, self.n_modes - 1, self.n_modes - 1)) # We set a general framework of n layers with (n-1)x(n-1) coeff.
        xtalk_mask[0, :, :] = 0     # In the first layer we act only on the even modes
        xtalk_mask[0, ::2, ::2] = 1
        xtalk_mask[-1, :, :] = 0    # In the last layer we act only on the odd ones
        xtalk_mask[-1, 1::2, 1::2] = 1
        xtalk_mask = sp.linalg.block_diag(*xtalk_mask) # Reshape into 2D array of bool

        return static_phases_mask, xtalk_mask


    def _generate_mixing_layers(self) -> Tuple[int, np.ndarray]:
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
        BS = np.array([[np.sqrt(trasm), 1j * np.sqrt(refl)],
                       [1j * np.sqrt(refl), np.sqrt(trasm)]])

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
        M = np.zeros((n_mixing_layers, self.n_modes, self.n_modes), dtype=complex)

        for ii in range(1, n_mixing_layers, 2):
            M[ii - 1, :, :] = V1 @ V2
            M[ii, :, :] = V2 @ V1
        M[0, :, :] = V1
        M[self.n_modes, :, :] = V2
        
        return n_mixing_layers, M 


    def U_from_phases(self, phases: np.ndarray) -> np.ndarray:
        """
        Calculate the 2D unitary matrix representation of the processor on the 
        basis of the provided phase array.
        
        Only n_modes-1 phases per layer are considered; this is sufficiently general
        because it defines the unitary apart from a global phase term. Supports 
        batch calculations (B, n_modes, n_modes) simultaneously.

        Parameters
        ----------
        phases : np.ndarray
            Array of optical phases in rad. Shape (n_phases,) or (B, n_phases).

        Returns
        -------
        np.ndarray
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
            phases = phases[np.newaxis, :]
            
        # Input validation
        if phases.shape[-1] != self.n_phases:
            raise ValueError(f"Dimension mismatch: expected {self.n_phases} elements, found {phases.shape[-1]}.")
        
        B = phases.shape[0]
        # (B, N, N) identity array
        U = np.tile(np.eye(self.n_modes, dtype=complex), (B, 1, 1)) 
                
        # Vectorized propagation
        for ii in range(self.n_mixing_layers - 1): 
            # Multiplication by the current mixing layer
            U = self.M[ii] @ U  
            
            # Extract phases for the current layer
            # Each layer has exactly (n_modes - 1) phases
            start_idx = ii * (self.n_modes - 1)
            end_idx = (ii + 1) * (self.n_modes - 1)
            layer_phases = phases[:, start_idx:end_idx]
            
            # Broadcasting: add a column axis to correctly multiply rows
            phase_multiplier = np.exp(1j * layer_phases)[:, :, np.newaxis]
            
            # Apply the phase shift only to the first (n_modes - 1) rows
            U[:, 0:(self.n_modes - 1), :] *= phase_multiplier
            
        # Last layer
        U = self.M[-1] @ U 
        
        return U[0] if is_1d else U        
    
    
    def phases_from_U(self, U_target: np.ndarray, max_iter: int = 100000, target_infidelity: float = 1e-6, plot_graph: bool = False) -> np.ndarray:
        """
        Given a unitary matrix U_target, finds the phases that implement it
        with the UPP. 
        
        Actually, to find the matrix more phases are used internally.
        Then, phases that are not used in the processor are simplified.
        The final matrix adheres to the target one only up to input/output
        phase terms. Fully supports batch processing for B matrices simultaneously.

        Parameters
        ----------
        U_target : np.ndarray
            Target unitary matrix. Shape (n_modes, n_modes) or (B, n_modes, n_modes).
        max_iter : int, optional
            Maximum number of iterations.
        target_infidelity : float, optional
            Target infidelity threshold to stop the optimization.
        plot_graph : bool, optional
            If True, plots the infidelity trend over iterations.
            
        Returns
        -------
        np.ndarray
            The array of optimal phases in rad. Shape (n_phases,) or (B, n_phases).
        """
        # Check dimension to guarantee vectorial shape
        is_2d = U_target.ndim == 2
        if is_2d:
            U_target = U_target[np.newaxis, ...]
        
        # Useful definitions
        B = U_target.shape[0]
        n_matrices = self.n_mixing_layers + 1  # Number of mixing matrices + the inverse of the target
        
        # Array of mixing matrices with inv_target at the end, over B batches
        v_matrices = np.zeros((B, n_matrices, self.n_modes, self.n_modes), dtype=complex)
        v_matrices[:, :-1] = self.M 
        v_matrices[:, -1] = U_target.conj().transpose(0, 2, 1)

        # Phase terms are introduced before each V matrix in this
        # decomposition; they are here initialized to random values
        phase_terms = np.exp(1j * np.random.rand(B, self.n_modes + 2, self.n_modes) * 2 * np.pi) 
        
        # Start the iterative decomposition
        print(f"Iterative phases optimization (Batch size: {B})")
        fidelity_monitor = np.zeros((B, max_iter))
        start_time = time.time()
        
        for step in range(max_iter):
            buffer_matrix = np.tile(np.eye(self.n_modes, dtype=complex), (B, 1, 1))
            
            # Calculates the matrix product
            for ii in range(n_matrices):  
                buffer_matrix = buffer_matrix * phase_terms[:, ii, :][:, :, np.newaxis] # First phases (multiply rows)
                buffer_matrix = v_matrices[:, ii, :, :] @ buffer_matrix  # Then mixing matrix
            
            # Vectorized fidelity check (Trace over inner axes)
            trace_b = np.trace(buffer_matrix, axis1=1, axis2=2)
            fid = np.abs(trace_b / self.n_modes)**2
            fidelity_monitor[:, step] = fid
            
            # If it is high enough it stops cycling
            if np.all((1 - fid) < target_infidelity):
                fidelity_monitor = fidelity_monitor[:, :step] 
                break
            
            if step > 0 and step % 1000 == 0:
                print(f"Iteration {step} - Max Infidelity: {np.max(1 - fid):.2e}")

            # If not, it performs another update cycle of all matrices            
            for ii in range(n_matrices):
                # Multiplication along columns to expose phases
                buffer_matrix = buffer_matrix * phase_terms[:, ii, :].conj()[:, np.newaxis, :] 

                # Update phase list                
                diagonal = np.diagonal(buffer_matrix, axis1=1, axis2=2).conj()
                phase_terms[:, ii, :] = diagonal / np.abs(diagonal) 
                
                # Multiplication along rows to apply new phases
                buffer_matrix = buffer_matrix * phase_terms[:, ii, :][:, :, np.newaxis] 
                
                # Removes next mixing layer
                buffer_matrix = v_matrices[:, ii, :, :] @ buffer_matrix @ v_matrices[:, ii, :, :].conj().transpose(0, 2, 1)
                
        print(f"--- Elapsed time: {(time.time() - start_time):.2f} seconds.")
        
        # Vectorized cleanup and restoration
        phase_terms = np.angle(phase_terms)     # Phases are now real numbers
        phase_terms = phase_terms - phase_terms[:, :, -1][:, :, np.newaxis]     # Subtract the phase of the last mode
        phase_terms = phase_terms[:, 1:-1, :]       # Remove external phases
        phase_terms[:, 0, ::2] -= phase_terms[:, 0, 1::2]       # First layer only even modes 
        phase_terms[:, 0, 1::2] = 0.0
        phase_terms[:, -1, 1:self.n_modes - 1:2] -= phase_terms[:, -1, 2:self.n_modes - 1:2]    # Last layer only odd modes
        phase_terms[:, -1, :self.n_modes - 1:2] = 0.0
        phase_terms = phase_terms[:, :, :-1]    # Removes last mode  
        phases = phase_terms.reshape(B, -1)
        phases = np.mod(phases, 2 * np.pi)
        
        # Check dimension in order to provide always squeezed shape
        if is_2d:
            phases = phases[0]
            fidelity_monitor = fidelity_monitor[0]

        # Output for debug
        if plot_graph:
            fig, ax = plt.subplots()
            ax.semilogy((1 - fidelity_monitor).T)
            ax.set_xlabel("Iteration number")
            ax.set_ylabel("Infidelity")
            ax.set_title("Decomposition convergence diagram")
            ax.grid(True, linestyle=':', alpha=0.6)
            plt.show()

        return phases


if __name__ == "__main__":
    
    from tools import fidelity, get_haar_unitary, make_real_border
        
    n = 6
    batch_size = 10
    processor = Bell(n_modes=n, calibration_file='./data/20260213_model_large_parameters_noBSerr.pickle') 
    processor = Bell(n_modes=n)

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

    # 5. Fidelity Calculation (tools.fidelity intrinsically supports batches!)
    fids = fidelity(u_target, u_exp)
    
    print("-" * 45)
    print(f"Final Fidelities: {fids}")
    
    if np.all(fids > 0.9999):
        print("Test PASSED: Batch decomposition and propagation are accurate.")
    else:
        print("Warning: Fidelity is below the expected threshold.")
    print("-" * 45)
