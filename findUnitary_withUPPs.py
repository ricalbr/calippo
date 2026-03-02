#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 13:11:46 2026

@author: andrea
"""

import numpy as np
#import scipy as sp

import UPPs

nModes=6

Utarget=UPPs.randomU(nModes) #extract random unitary

bellone=UPPs.Bell(nModes)

bellone.phasesFromU(Utarget,plotGraph=True)

Udone=bellone.U[0,:,:] #extracts U
Udone=UPPs.makeRealBorder(Udone)[0] #make real bordered

UtargetR=UPPs.makeRealBorder(Utarget)[0] #also the target one

fidelityFinal=np.abs(np.sum(Udone*UtargetR.conj()))/nModes #and compares them

print("Checking real-bordered matrices we have reached fidelity ",fidelityFinal)
