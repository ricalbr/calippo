#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 17:51:22 2026

@author: andrea
"""

import numpy as np
import matplotlib.pyplot as plt
import UPPs

# Sets up a 6x6 Bell processor
bellone = UPPs.Bell(6)

nU = 200  # this is the number of unitaries that will be used in this script

# ZEROTH STEP
# Generates a set of Coefficients to be taken as "true" coefficients of
# the device
A = bellone.generateTestCoefficients(gain0=3, amplPhi0=1)

# FIRST STEP ----------------------------------------------------------------
# Simulates the generation of nU unitaries with the "true" device
#
I2 = bellone.generateRandomCurrents(nU) * np.pi  # extracts a set of currents
bellone.phasesFromCurrAndA(I2, A)  # calculates the internal phases phi given by
# those currents and the coeff. A, storing them
# in the UPP object
bellone.calcU()
setOfUAmpl = np.abs(bellone.U)  # Calculates the unitaries that correspond to those
# phases stored in the UPP modulus
# takes the amplitude (dropping the phase terms)
# these are stored in "setOfUAmpl"

# SECOND STEP ----------------------------------------------------------------
# Runs the calibration procedure
#

# This is to initialize the minimization
A0 = bellone.generateTestCoefficients(gain0=3.5, amplPhi0=1)
# Checks for future comparison the fidelity that is reached with this A0, for the
# currents A0
bellone.phasesFromCurrAndA(I2, A0)
bellone.calcU()
print()
print(
    "- Fidelity before optimization on the learning set: ",
    bellone.amplitudeFidelityWith(setOfUAmpl),
)
print()

Afit = bellone.calibrateUPP(setOfUAmpl, I2, A0)  # calibration command

bellone.Astored = Afit  # stores the calibration in the UPP

# Sets the phases of the UPP as the results of the fit and checks the fidelity now
bellone.phasesFromCurrents(I2)
print(
    "Fidelity after optimization, on the learning set: ",
    bellone.amplitudeFidelityWith(setOfUAmpl),
)

plt.figure(1)
plt.clf()
# plt.plot(A0,label='start')
plt.plot(A, "-", label="true")
plt.plot(Afit, "x", label="fit")
plt.legend()

# THIRD STEP ----------------------------------------------------------------
# Checks the calibration against other nU unitaries
I2 = bellone.generateRandomCurrents(nU) * np.pi  # extracts other nU sets of currents
bellone.phasesFromCurrAndA(I2, A)
bellone.calcU()
setOfUAmpl = np.abs(
    bellone.U
)  # Unitaries (without phases) calculated with true A coefficients

bellone.phasesFromCurrents(I2)
bellone.calcU()
print(
    "Fidelity after optimization, on a further test set: ",
    bellone.amplitudeFidelityWith(setOfUAmpl),
)  # Fidelity with unitaries calculated with A coefficients from fit
