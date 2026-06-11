#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 17:51:22 2026

@author: andrea
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import UPPs


#FIRST STEP
#Load data and initializes processor

#Load data from file

with open('20260210_BellOne_ML_randpowers.bin', 'rb') as f:
    DATA = pickle.load(f)
    
setOfUAmpl = DATA['Us'] #Data of the unitary matrices
nU=setOfUAmpl.shape[0]
n=np.sqrt(setOfUAmpl.shape[1]).astype('int')
setOfUAmpl=setOfUAmpl.reshape(nU,n,n)

I2 = ( DATA['currents'] )**2


#Sets up a 6x6 Bell processor
bellone=UPPs.Bell(n)




# SECOND STEP ----------------------------------------------------------------
# Runs the calibration procedure
#

#This is to initialize the minimization
A0=bellone.generateTestCoefficients(gain0=0.07,amplPhi0=0,noise0=0)
#Checks for future comparison the fidelity that is reached with this A0, for the
#currents A0
bellone.phasesFromCurrAndA(I2,A0)
bellone.calcU()
print()
print("- Fidelity before optimization on the learning set: ",bellone.amplitudeFidelityWith(setOfUAmpl))
print()

Afit = bellone.calibrateUPP(setOfUAmpl,I2,A0) #calibration command

bellone.Astored=Afit #stores the calibration in the UPP

#Sets the phases of the UPP as the results of the fit and checks the fidelity now
bellone.phasesFromCurrents(I2)
print("Fidelity after optimization, on the learning set: ",bellone.amplitudeFidelityWith(setOfUAmpl))



# THIRD STEP
# Compares with Ciro's calibration
bellone2=UPPs.Bell(n)

# 1. Loads calibration parameters: static phases and cross-talk matrix.
UPPs.importEphosCalibration(bellone2,'20260213_model_large_parameters_noBSerr.pickle')

compareA=bellone.Astored-bellone2.Astored

print("Compares with Ciro's calibration")
print("Maximum difference on cross-talk coefficients")
print(np.max(np.abs(compareA[:(n*(n-1)**2)])))
print("Maximum difference on phases at rest")
deltaPhi0=compareA[(n*(n-1)**2):]
deltaPhi0[deltaPhi0<-np.pi]=deltaPhi0[deltaPhi0<-np.pi]+2*np.pi
print(np.max(np.abs(deltaPhi0)))



# plt.figure(1)
# plt.clf()
# #plt.plot(A0,label='start')
# plt.plot(A,'-',label='true')
# plt.plot(Afit,'x',label='fit')
# plt.legend()

# # THIRD STEP ----------------------------------------------------------------
# # Checks the calibration against other nU unitaries
# I2 = bellone.generateRandomCurrents(nU)*np.pi #extracts other nU sets of currents
# bellone.phasesFromCurrAndA(I2,A)
# bellone.calcU()
# setOfUAmpl=np.abs(bellone.U) #Unitaries (without phases) calculated with true A coefficients

# bellone.phasesFromCurrents(I2)
# bellone.calcU()
# print("Fidelity after optimization, on a further test set: ",bellone.amplitudeFidelityWith(setOfUAmpl)) #Fidelity with unitaries calculated with A coefficients from fit




