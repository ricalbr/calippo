#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 23:05:56 2026

@author: Francesco Ceccarelli

Control module for Universal Photonic Processors (UPP).

This module provides a high-level class to automate the characterization of
multi-mode universal photonic processors, including optical transmission,
unitary matrix measurement, and phase shifter resistance characterization.

Classes
-------
UniversalPhotonicProcessor
    Main controller for UPP hardware orchestration (Switch, OPM, Qontrol).
"""

from typing import List, Optional

import numpy as np

# Hardware imports
from fastruments.LfiberSwtich import FiberSwitch
from fastruments.Qontrol import Q8iv
from fastruments.SantecOPM150 import OPM150

from calippo.model import Bell, Clements

# UPP imports
from calippo.tools import sinkhorn


class UniversalPhotonicProcessor:
    """
    Controller for a multi-mode Universal Photonic Processor (UPP).

    Integrates an Lfiber Switch, a Santec OPM150 and a Qontrol Q8iv driver
    to provide full characterization of the photonic chip. It also initializes
    a digital twin based on the specified mesh architecture.

    Parameters
    ----------
    num_modes : int
        Number of optical modes (dimension N).
    mesh_type : str
        Topology of the photonic mesh (e.g., 'bell', 'clements').
        Used to instantiate the correct digital twin.
    wavelength_nm : int
        Operation wavelength in nanometers.
    imax : float
        Current compliance in milliamperes.
    vmax : float
        Voltage compliance in volts.
    switch_resource : str, optional
        VISA resource string for the Lfiber switch. If provided, initializes switch and OPM.
    q8iv_resource : str, optional
        Serial port for the Qontrol driver. If provided, initializes the driver.
    switch_channels : List[int], optional
        Physical switch ports connected to UPP inputs. Default is 1 to N.
    opm_channels : List[int], optional
        Physical OPM ports connected to UPP outputs. Default is 1 to N.
    shifter_channels : List[int], optional
        Physical Q8iv channels connected to the phase shifters. Default is 0 to N(N-1)-1.
    switch_transient : float, optional
        Wait time (s) after switching mode. Default is 0.1.
    UPP_transient : float, optional
        Wait time (s) after UPP reconfiguration. Default is 0.5.
    verbose : bool, optional
        If True, prints status messages. Default is True.

    Raises
    ------
    ValueError
        If mapping lists do not match expectations.
    ValueError
        If an unsupported mesh type is provided.
    """

    _MESHES_ = {"clements": Clements, "bell": Bell}

    def __init__(
        self,
        num_modes: int,
        mesh_type: str,
        wavelength_nm: int,
        imax: float,
        vmax: float,
        switch_resource: Optional[str] = None,
        q8iv_resource: Optional[str] = None,
        switch_channels: Optional[List[int]] = None,
        opm_channels: Optional[List[int]] = None,
        shifter_channels: Optional[List[int]] = None,
        switch_transient: float = 0.5,
        UPP_transient: float = 0.5,
        verbose: bool = True,
    ):
        self.num_modes = num_modes

        # Input validation for MZI mesh type
        if mesh_type.lower() not in self._MESHES_:
            raise ValueError(
                f"[UPP][ERROR] Unsupported mesh_type: '{mesh_type}'. Available options: {list(self._MESHES_.keys())}"
            )
        self.dtwin = self._MESHES_[mesh_type.lower()](num_modes)

        self.imax = imax
        self.vmax = vmax
        self.verbose = verbose
        self.switch_transient = switch_transient
        self.UPP_transient = UPP_transient

        # Setup optical and electrical mapping
        self.input_map = switch_channels if switch_channels else list(range(1, num_modes + 1))
        self.output_map = opm_channels if opm_channels else list(range(1, num_modes + 1))
        self.shifter_map = shifter_channels if shifter_channels else list(range(0, num_modes * (num_modes - 1)))

        if len(self.input_map) != num_modes or len(self.output_map) != num_modes:
            raise ValueError("[UPP][ERROR] Optical mapping lists must match number of modes N.")
        if len(self.shifter_map) != num_modes * (num_modes - 1):
            raise ValueError("[UPP][ERROR] Electrical mapping lists must match number of shifters N(N-1).")

        # Initialize Hardware 1 & 2: Fiber Switch and Power Meter (Optional)
        if switch_resource:
            self.switch = FiberSwitch(switch_resource, transient=self.switch_transient, verbose=verbose)
            self.pm = OPM150(power_unit=1, verbose=verbose)
            self.pm.wavelength = wavelength_nm
            self.pm.autorange_all(True)
        else:
            self.switch = None
            self.pm = None

        # Initialize Hardware 3: Qontrol Driver (Optional)
        if q8iv_resource:
            self.drv = Q8iv(
                q8iv_resource,
                init_mode="i",
                imax=self.imax,
                vmax=self.vmax,
                transient=self.UPP_transient,
                verbose=verbose,
            )
        else:
            self.drv = None

    def measure_power_matrix(self) -> np.ndarray:
        """
        Measure the raw NxN optical power matrix.

        Returns
        -------
        np.ndarray
            NxN array where the element (j, i) represents the optical power
            (watts) from input i to output j.

        Raises
        ------
        RuntimeError
            If optical hardware (switch/OPM) has not been initialized.
        """
        if not self.switch or not self.pm:
            raise RuntimeError("[UPP] Optical hardware (switch/OPM) not initialized.")
        if self.verbose:
            print(f"[UPP] Measuring the transmission on the {self.num_modes} input modes...")
        power_matrix = np.zeros((self.num_modes, self.num_modes))
        for i, ch_in in enumerate(self.input_map):
            self.switch.set_channel(ch_in)
            column_powers = self.pm.read_power(channels=self.output_map)
            power_matrix[:, i] = np.array(column_powers)
        power_matrix[power_matrix < 0] = 0
        return power_matrix

    def measure_unitary_matrix(self) -> np.ndarray:
        """
        Measure the power matrix, apply Sinkhorn normalization and return
        square root values.

        Returns
        -------
        np.ndarray
            NxN array of the reconstructed unitary amplitudes |U_ij|.

        Raises
        ------
        RuntimeError
            If optical hardware (switch/OPM) has not been initialized.
        """
        if not self.switch or not self.pm:
            raise RuntimeError("[UPP][ERROR] Optical hardware (switch/OPM) not initialized.")
        power_matrix = self.measure_power_matrix()
        if self.verbose:
            print("[UPP] ... Normalizing...")
        normalized_power, _, _ = sinkhorn(power_matrix)
        return np.sqrt(normalized_power)

    def measure_shifter_resistances(self, test_current: float = 2.0) -> np.ndarray:
        """
        Measure the electrical resistance of the phase shifters.

        Parameters
        ----------
        test_current : float, optional
            Current in mA applied to measure resistance. Default is 2.0.

        Returns
        -------
        np.ndarray
            1D array of measured resistances (ohms) for each phase shifter.

        Raises
        ------
        RuntimeError
            If electrical hardware (Qontrol) has not been initialized.
        """
        if not self.drv:
            raise RuntimeError("[UPP][ERROR] Electrical hardware (Qontrol) not initialized.")
        if self.verbose:
            print(f"[UPP] Measuring resistance for {len(self.shifter_map)} shifters...")
        try:
            self.drv.set_current(self.shifter_map, [test_current] * len(self.shifter_map))
            resistances = (
                np.array(self.drv.voltage)[self.shifter_map] / np.array(self.drv.current)[self.shifter_map]
            ) * 1000.0
        finally:
            self.drv.set_all_zero()
        if self.verbose:
            print("[UPP] ... Done!")
        return resistances

    def set_shifter_currents(self, currents: np.ndarray):
        """
        Apply a vector of currents to all mapped phase shifters.

        Parameters
        ----------
        currents : np.ndarray
            1D array or list of current values (mA) to be applied.
            The length must match the number of shifters N(N-1).

        Raises
        ------
        RuntimeError
            If electrical hardware (Qontrol) has not been initialized.
        """
        if not self.drv:
            raise RuntimeError("[UPP] Electrical hardware (Qontrol) not initialized.")
        self.drv.set_current(self.shifter_map, currents)

    def set_all_zero(self):
        """
        Set all phase shifter currents to zero.

        Raises
        ------
        RuntimeError
            If electrical hardware (Qontrol) has not been initialized.
        """
        if not self.drv:
            raise RuntimeError("[UPP] Electrical hardware (Qontrol) not initialized.")
        self.drv.set_all_zero()

    def close(self):
        """
        Release only initialized hardware resources.
        """
        if self.drv is not None:
            self.drv.close()
        if self.switch is not None:
            self.switch.close()
        if self.pm is not None:
            self.pm.close()
