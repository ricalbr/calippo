#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 16:07:49 2026

@author: Andrea Crespi, Francesco Ceccarelli

Digital twin models for Universal Photonic Processors (UPP).

This module defines objects implementing a photonic processor built 
according to a sequence of mixing and phase layers. Multiple instances 
are supported in parallel, namely multiple configurations of phases 
can be calculated together.

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
    It defines the common interface that all mesh topologies must implement.
    
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
            
        self.n_modes: int = n_modes                                  # Number of modes
        self.n_phases: int = (n_modes - 1) * n_modes                 # Number of tunable parameters
        self.n_used_phases: int = (n_modes - 1) * (n_modes - 1)      # Number of tunable parameters actually used in the experiments
                        
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
        with open(fileName, 'rb') as f:
            calibration = pickle.load(f)

        if calibration['N'] != self.n_modes:
            raise ValueError(f"Calibration size mismatch: expected {self.n_modes} modes, "
                             f"found {calibration['N']}.")
            
        if calibration['n_heaters'] != self.n_phases:
            raise ValueError(f"Calibration size mismatch: expected {self.n_phases} phases, "
                             f"found {calibration['n_heaters']}.")

        static_phases = np.mod(calibration['static_phases'], 2 * np.pi)
        static_phases = static_phases * self.static_phases_mask
        xtalk = calibration['xt_matrix']
        xtalk = xtalk * self.xtalk_mask

        return static_phases, xtalk
        

    def generate_random_calibration(self, xtalk_ampl: float, static_phases_ampl: float, noise: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a random calibration matrix for simulation purposes.
        
        On average, current-squared-to-phase gain is driven by `xtalk_ampl`, 
        and cross-talks are close to 0. Phases at rest are randomly distributed.

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
        if noise is None:
            noise = xtalk_ampl / 10.0
            
        static_phases = np.mod(np.random.rand(self.n_phases) * static_phases_ampl, 2 * np.pi)
        static_phases = static_phases * self.static_phases_mask
        xtalk = np.eye(self.n_phases) * xtalk_ampl + (np.random.rand(self.n_phases, self.n_phases) - 0.5) * noise
        xtalk = xtalk * self.xtalk_mask

        return static_phases, xtalk


    def generate_random_currents(self, current_ampl: float) -> np.ndarray:
        """
        Generate a 1D array of random currents in mA.
        
        The square of the output is uniformly distributed between 0 and 1,
        then scaled by the provided maximum amplitude. Unused phases according
        to the internal mask are strictly kept at zero.

        Parameters
        ----------
        current_ampl : float
            Maximum current amplitude in mA.

        Returns
        -------
        np.ndarray
            Array of shape (n_phases,) with randomized currents in mA.
        """
        I = np.sqrt(np.random.rand(self.n_phases)) * current_ampl
        I *= self.static_phases_mask
        return I


    def phases_from_currents(self, currents: np.ndarray) -> np.ndarray:
        """
        Calculate the optical phases for a 1D vector of currents based on the 
        cross-talk coefficients stored internally.

        Parameters
        ----------
        currents : np.ndarray
            1D array of currents of shape (n_phases,) in mA.

        Returns
        -------
        np.ndarray
            Array of calculated phases in rad.
        """
        # Input validation
        currents = np.squeeze(currents)
        if currents.size != self.n_phases:
            raise ValueError(
                f"Dimension mismatch: expected {self.n_phases} elements, "
                f"but 'currents' has size {currents.size}."
            )
            
        # Calculate phases
        phases = self.static_phases + self.xtalk @ currents ** 2
        return np.mod(phases, 2 * np.pi)
    

    @abstractmethod
    def U_from_phases(self, phases: np.ndarray) -> np.ndarray:
        pass
        
    
    def currents_from_phases(self, phases: np.ndarray) -> np.ndarray:
        """
        Calculate the currents in mA required to achieve target optical phases,
        compensating for cross-talk using the calibration matrix. 
        
        The method solves the linear system for squared currents and iteratively 
        adjusts phases by 2*pi if negative squared currents are found.

        Parameters
        ----------
        phases : np.ndarray
            1D array of target phase values in rad.

        Returns
        -------
        np.ndarray
            1D array of currents in mA.
        """
        # Input validation
        phases= np.mod(phases, 2 * np.pi)
        phases = np.squeeze(phases)
        if phases.size != self.n_phases:
            raise ValueError(
                f"Dimension mismatch: expected {self.n_phases} elements, "
                f"but 'phases' has size {phases.size}."
            )

        # Reduce dimensions to used parameters in order to avoid singularities
        phases = phases[self.static_phases_mask]
        static_phases = self.static_phases[self.static_phases_mask]
        used_rows = np.any(self.xtalk_mask, axis=0)
        xtalk = self.xtalk[np.ix_(used_rows, used_rows)]

        # Numerical inversion
        I2 = np.linalg.solve(xtalk, phases - static_phases)
        while np.any(I2 < 0):
            phases[I2 < 0] += 2 * np.pi
            I2 = np.linalg.solve(xtalk, phases - static_phases)
        
        # Reconstruct original dimensions
        I = np.zeros_like(self.static_phases_mask, dtype=float)
        I[self.static_phases_mask] = np.sqrt(I2)
        
        return I


    @abstractmethod
    def phases_from_U(self, U_target: np.ndarray, max_iter: int = 100000, target_infidelity: float = 1e-5, plot_graph: bool = False) -> np.ndarray:
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


    def phases_from_U(self, U_target: np.ndarray, max_iter: int = 100000, target_infidelity: float = 1e-5, plot_graph: bool = False) -> np.ndarray:        
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
    refl : float, optional
        Power reflectivity of the basic beam splitter. Default is 0.5.
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
        architecture.

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
        basis of the provided 1D phase array.
        
        Only n-1 phases per layer are considered; this is sufficiently general
        because it defines the unitary apart from a global phase term.

        Parameters
        ----------
        phases : np.ndarray
            1D array of optical phases in rad.

        Returns
        -------
        np.ndarray
            The 2D complex unitary matrix resulting from the propagation.
        """
        # Input validation
        phases = np.squeeze(phases)
        if phases.size != self.n_phases:
            raise ValueError(
                f"Dimension mismatch: expected {self.n_phases} elements, "
                f"but 'phases' has size {phases.size}."
            )
        
        # Prepare an array containing the identity matrix
        U = np.eye(self.n_modes, dtype=complex) 
                
        # Propagation
        for ii in range(self.n_mixing_layers - 1): 
            # Multiplication by the current mixing layer
            U = self.M[ii] @ U  
            
            # Extract phases for the current layer
            # Each layer has exactly (n_modes - 1) phases
            start_idx = ii * (self.n_modes - 1)
            end_idx = (ii + 1) * (self.n_modes - 1)
            layer_phases = phases[start_idx:end_idx]
            
            # Transform the 1D array into a 2D column vector (n_modes - 1, 1)
            # This allows broadcasting to correctly multiply the rows of U
            phase_multiplier = np.exp(1j * layer_phases)[:, np.newaxis]
            
            # Apply the phase shift only to the first (n_modes - 1) rows
            U[0:(self.n_modes - 1), :] = U[0:(self.n_modes - 1), :] * phase_multiplier
            
        # Last layer
        U = self.M[-1] @ U 
        
        return U        
    
    
    def phases_from_U(self, U_target: np.ndarray, max_iter: int = 100000, target_infidelity: float = 1e-5, plot_graph: bool = False) -> np.ndarray:
        """
        Given a unitary matrix U_target, finds the phases that implement it
        with the UPP. 
        
        Actually, to find the matrix more phases are used internally.
        Then, phases that are not used in the processor are simplified.
        The final matrix adheres to the target one only up to input/output
        phase terms.

        Parameters
        ----------
        U_target : np.ndarray
            Target unitary matrix.
        max_iter : int, optional
            Maximum number of iterations.
        target_infidelity : float, optional
            Target infidelity.
        plot_graph : bool, optional
            If True, plots the infidelity trend over iterations.
            
        Returns
        -------
        np.ndarray
            The 1D array of optimal phases in rad.
        """
        print("Iterative phases optimization to find the target unitary.")
        
        n_matrices = self.n_mixing_layers + 1  # Number of mixing matrices + the inverse of the target
        v_matrices = np.zeros((n_matrices, self.n_modes, self.n_modes), dtype=complex)  # Array of matrices
        v_matrices[:-1] = self.M  # The V matrices are the mixing matrices of the UPP
        v_matrices[-1] = U_target.conj().T  # Except from the last one which is the inverse of the target

        # Phase terms are introduced before each V matrix in this
        # decomposition; they are here initialized to random values
        phase_terms = np.exp(1j * np.random.rand(self.n_modes + 2, self.n_modes) * 2 * np.pi) 
        
        fidelity_monitor = np.zeros(max_iter)
        
        start_time = time.time()
        
        for step in range(max_iter):
          
            buffer_matrix = np.eye(self.n_modes, dtype=complex)
            
            for ii in range(n_matrices):  # Calculates the matrix product
                buffer_matrix = buffer_matrix * (phase_terms[ii, :])[:, np.newaxis]  # First phases (multiply rows)
                buffer_matrix = v_matrices[ii, :, :] @ buffer_matrix  # Then mixing matrix
            
            # Checks fidelity
            fidelity_monitor[step] = np.abs(np.trace(buffer_matrix) / self.n_modes)**2
            
            # If it is high enough it stops cycling
            if (1 - fidelity_monitor[step]) < target_infidelity:
                fidelity_monitor = fidelity_monitor[:step]  # Truncates the unused part of the array
                break
            
            if step % 1000 == 0:
                print(f"Iteration {step}")
            
            # If not, it performs another update cycle of all matrices
            for ii in range(n_matrices):
                # Expose phases
                buffer_matrix = buffer_matrix * phase_terms[ii, :].conj()  # Multiply columns
                
                # Update phase list
                diagonal = np.diag(buffer_matrix).conj()
                phase_terms[ii, :] = diagonal / np.abs(diagonal) 
                
                # Applies new phases
                buffer_matrix = buffer_matrix * (phase_terms[ii, :])[:, np.newaxis]  # Multiply rows 
                
                # Removes next mixing layer
                buffer_matrix = v_matrices[ii, :, :] @ buffer_matrix @ v_matrices[ii, :, :].conj().T
                
        # End of optimization      
        print(f"--- Elapsed time: {(time.time() - start_time):.2f} seconds.")
        
        # Brings phases back into the UPP; from now on phase_terms contains
        # phases as real numbers
        phase_terms = np.angle(phase_terms)
        phase_terms = phase_terms - phase_terms[:, -1, np.newaxis]  # Subtracts the phase of the last mode
        phase_terms = phase_terms[1:-1, :]  # Remove external phases
        phase_terms[0, ::2] -= phase_terms[0, 1::2]  # First layer only even modes 
        phase_terms[0, 1::2] = 0.0
        phase_terms[-1, 1:self.n_modes - 1:2] -= phase_terms[-1, 2:self.n_modes - 1:2]  # Last layer only odd modes
        phase_terms[-1, :self.n_modes - 1:2] = 0.0
        phase_terms = phase_terms[:, :-1]  # Removes last mode        
        phases = phase_terms.flatten()
        phases = np.mod(phases, 2 * np.pi)
        
        # Output for debug
        if plot_graph:
            plt.figure()
            plt.clf()
            plt.semilogy(1 - fidelity_monitor)
            plt.xlabel("Iteration number")
            plt.ylabel("Infidelity")
            
        return phases

        
if __name__ == "__main__":
    from tools import fidelity, get_haar_unitary, make_real_border  # Import tools for UPPs
    
    try:
        # Configuration
        n = 6
        #processor = Bell(n_modes=n, calibration_file='./bkp/20260213_model_large_parameters_noBSerr.pickle') 
        processor = Bell(n_modes=n)

        # 1. Target: Generate a random Haar unitary and make it "real-bordered"
        u_target_raw = get_haar_unitary(n)
        u_target, _, _ = make_real_border(u_target_raw)

        # 2. Decomposition: Solve for target phases and then for required currents
        phases_target = processor.phases_from_U(u_target)
        currents_target = processor.currents_from_phases(phases_target)

        # 3. Forward Propagation: From currents, retrieve phases and compute final unitary
        # Note: phases_from_currents accounts for internal cross-talk
        phases_exp = processor.phases_from_currents(currents_target)
        u_exp_raw = processor.U_from_phases(phases_exp)
        
        # 4. Real-border the experimental unitary for a consistent fidelity comparison
        u_exp, _, _ = make_real_border(u_exp_raw)

        # 5. Calculate Fidelity
        fid = fidelity(u_target, u_exp)
        
        print("-" * 45)
        print("--- Decomposition and Forward Propagation Test ---")
        print(f"Modes: {n}")
        print(f"Final Fidelity: {fid:.6f}")
        
        # Validation check
        if fid > 0.9999:
            print("Test PASSED: Decomposition and propagation are accurate.")
        else:
            print("Warning: Fidelity is below the expected threshold.")
        print("-" * 45)

    except Exception as e:
        print(f"An error occurred during the test: {e}")