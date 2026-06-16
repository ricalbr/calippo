#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 20:55:58 2026

@author: andrea
"""

import numpy as np
import UPPs

# 0. Setting up a 6-modes processor
bellone = UPPs.Bell(6)

# 1. Loads calibration parameters: static phases and cross-talk matrix.
UPPs.importEphosCalibration(bellone, "20260213_model_large_parameters_noBSerr.pickle")

# 2. Generate target phases for active heaters
phiInitial = np.zeros((1, bellone.nPhases))
phiInitial[:, bellone.Pmask] = (
    np.random.rand(1, bellone.nCurrents) * 2 * np.pi
)  # uniform(0, 2 * np.pi, bellone.nCurrents)


# 3. Inverts the cross-talk matrix with phase unwrapping (phi + 2*pi*n).
bellone.setThesePhases(phiInitial)
I2 = bellone.currentsFromPhases()  # squared currents
I_full = np.sqrt(I2)  # currents

# 4. Validates the inversion against the physical model.
bellone.phasesFromCurrents(I2)
phiReconstructed = bellone.phi
error_full = (phiReconstructed - phiInitial + np.pi) % (2 * np.pi) - np.pi

# 5. Print results.
print("-" * 30)
print("MODEL INVERSION")
print("-" * 30)
print(f"Total number of heaters: {bellone.nPhases}")
print(f"Active heaters: {bellone.nCurrents}")
print(f"Max phase error: {np.max(np.abs(error_full)):.2e} rad")
print("\nFinal current vector (mA):")
print(I_full)
