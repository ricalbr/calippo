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

Functions
---------
importEphosCalibration(UPPobject, fileName)
    Import calibration data from a pickle file.
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time
import pickle
from abc import ABC, abstractmethod
from .tools import fidelity, amplitude_fidelity

###############################################################################
#                           BASE ABSTRACT CLASS                               #
###############################################################################

class Mesh(ABC):
    """
    Abstract base class for Universal Photonic Processors.
    It defines the common interface that all mesh topologies must implement.
    """

    @abstractmethod
    def generateTestCoefficients(self, gain0, amplPhi0, noise0=None):
        pass

    @abstractmethod
    def generateRandomCurrents(self, nU_temp):
        pass

    @abstractmethod
    def calcU(self):
        pass

    @abstractmethod
    def phasesFromCurrAndA(self, I2, A):
        pass

    @abstractmethod
    def phasesFromCurrents(self, I2):
        pass

    @abstractmethod
    def currentsFromPhases(self):
        pass

    @abstractmethod
    def setThesePhases(self, phi_full):
        pass

    @abstractmethod
    def phasesFromU(self, Utarget, maxIter=100000, tarInf=1e-10, plotGraph=False):
        pass

    @abstractmethod
    def calibrateUPP(self, setOfUAmpl, I2, A0):
        pass

    def amplitudeFidelityWith(self, Uref):
        """
        Calculates the mean amplitude fidelity between the set of unitaries 
        stored in the object and a set of unitaries of reference Uref.
        """
        fids = amplitude_fidelity(Uref, self.U)
        return np.mean(fids)
    
    def fidelityWith(self, Uref):
        """
        Calculates the mean fidelity between the set of unitaries 
        stored in the object and a set of unitaries of reference Uref.
        """
        fids = fidelity(Uref, self.U)
        return np.mean(fids)


###############################################################################
#                               CLASS CLEMENTS                                #
###############################################################################
 
class Clements(Mesh):
    """
    A photonic processor built according to the Clements scheme.
    """
    def __init__(self, nModes, Refl=0.5):
        pass
        
    def generateTestCoefficients(self, gain0, amplPhi0, noise0=None):
        pass

    def generateRandomCurrents(self, nU_temp):
        pass

    def calcU(self):
        pass

    def phasesFromCurrAndA(self, I2, A):
        pass

    def phasesFromCurrents(self, I2):
        pass

    def currentsFromPhases(self):
        pass

    def setThesePhases(self, phi_full):
        pass

    def phasesFromU(self, Utarget, maxIter=100000, tarInf=1e-10, plotGraph=False):
        pass

    def calibrateUPP(self, setOfUAmpl, I2, A0):
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
    
    """
    def __init__(self, nModes, Refl=0.5):
        """
        The initialization method of the class. One can specify the number of 
        modes (nModes, needed parameter) and the power reflectivity of the 
        basic beam splitter (Refl, optional parameter). If the power reflectivity
        is not specified, Refl = 1/2 is assumed.
        Only EVEN number of modes are at present supported.
        """
        if nModes % 2 ==1:
            nModes=nModes-1
            print("Modes must be EVEN for this version.")
            print(f"We shall take {nModes} modes.")
            
        self.n=nModes #the number of modes must be defined
        self.nM=self.n+1 #number of mixing layers (EXCLUDING the phase blocks)
        
        self.nU=1 #to begin with, we set the number of contained unitaries
                    #equal to 1
                    

        #phi is the constitutive parameters that determines the built-in unitary
        #it contains nU sets of n vectors of length n-1, i.e. an array nU x n(n-1)
        #where nU is the number of unitaries to be calculated at the same time
        self.nPhases=self.n*(self.n-1)
        self.phi=np.zeros((self.nU,self.nPhases))

        #NOW WE CALCULATE THE MIXING LAYERS        
        #matrix of two-mode beam splitter, which is the basic block
        Trasm=1-Refl
        BS = np.array([[np.sqrt(Trasm), 1j*np.sqrt(Refl)],
                       [1j*np.sqrt(Refl), np.sqrt(Trasm)]])

        #first kind of beam-splitter layer (V1)
        V1=BS
        for ii in range(1,int(self.n/2)):
            V1=sp.linalg.block_diag(V1,BS)

        #second kind of beam-splitter layer (V2)
        V2=np.array([[1]]) 
        for ii in range(1,int(self.n/2)):
            V2=sp.linalg.block_diag(V2,BS)
        V2=sp.linalg.block_diag(V2, np.array([[1]]) )

        #array of matrices that describe the processor (mixing layers)
        self.M=np.zeros((self.nM,self.n,self.n),dtype=complex)

        for ii in range(1,self.nM,2):
            self.M[ii-1,:,:]=V1@V2
            self.M[ii,:,:]=V2@V1
        self.M[0,:,:]=V1
        self.M[self.n,:,:]=V2
            
        self.calcU() #Executes the method calcU() which calculates the unitaries
                    #associated to phi and stores them in the object
        
        #NOW WE CALCULATE THE MASKS
        #that relate the actual free parameters of calibration
        #to the available ones

        #Mask for phases and currents
        self.Pmask=np.ones((self.n,self.n-1)) #we have n(n-1) overall phases
        self.Pmask[0,:]=0        #In the first layer we act only on the even ones
        self.Pmask[0,::2]=1
        self.Pmask[-1,:]=0       #In the last layer we act only on the odd ones
        self.Pmask[-1,1::2]=1
        self.Pmask=self.Pmask.reshape(-1).astype('bool') #reshape into 1D array of bool
        #Mask for cross-talk coefficients
        self.Cmask=np.ones((self.n,self.n-1,self.n-1))  #We set a general framework of n layers with (n-1)x(n-1) coeff.
        self.Cmask[0,:,:]=0    #In the first layer we act only on the even modes
        self.Cmask[0,::2,::2]=1
        self.Cmask[-1,:,:]=0 #In the last layer we act only on the odd ones
        self.Cmask[-1,1::2,1::2]=1
        self.Cmask=self.Cmask.reshape(-1).astype('bool') #reshape into 1D array of bool

        self.Amask=np.concatenate((self.Cmask,self.Pmask))

        self.nCurrents=np.sum(self.Pmask)
        self.nTotCoeff=np.sum(self.Amask)
        
        #A set of cross-talk coefficients and phi0 
        #is also stored in the processor; for initialization
        #we use a trivial set.
        self.Astored=self.generateTestCoefficients(1, 0, 0)
        
        
        print("A new processor was created")
        print(f"This has {self.n} modes, {self.nM} mixing layers, {self.nPhases} phases.")
        print(f"However, only {self.nCurrents} phase terms are directly actuated by currents")
        print(f"so the free parameters including cross-talks are {self.nTotCoeff}.")
        print()
        print("Used beam-splitter matrix")
        print(BS)
        print()
        print()
        
 
    
    def generateTestCoefficients(self,gain0,amplPhi0,noise0=None):
        """
        Returns a random array of coefficients A. On average, 
        currentSquared-to-Phase is gain0, cross-talks equal to 0.
        Noise is added on gains and cross-talks for an amount equal to
        noise0. If noise0 is not specified, noise0=gain0/10 is taken.
        Phases at rest are randomly distributed between 0 to amplPhi0.
        """
        if noise0==None:
            noise0=gain0/10
            
        Cflatted=np.tile(np.eye(self.n-1). 
                         reshape(-1) * gain0, self.n ) + np.random.rand(
                             self.n*(self.n-1)**2)*noise0
        phi0=np.random.rand(self.n*(self.n-1))*amplPhi0
        Cflatted=Cflatted*self.Cmask #unrelevant coefficients set to zero
        phi0=phi0*self.Pmask #unrelevant coefficients set to zero
        
        return np.hstack((Cflatted, phi0))

    def generateRandomCurrents(self,nU_temp):
        """
        Generates an array consisting of nU vectors of random currents squared
        between 0 and 1; the vector length is nPhases but only 
        the relevant phases are non-zero.
        
        """                              #only to feed relevant phases
        I2 = np.random.rand(nU_temp,self.nPhases)
        I2 = I2*self.Pmask #only the relevant ones are kept
        
        return I2
    
    def calcU(self):
       """
       Carry on the calculation of the unitaries on the basis of the stored
       phases phi. These unitaries are 3D array, namely a stack of matrices: 
       first dimension is the index  of the unitary, then we have rows 
       and columns. Unitaries are stored in the object as self.U.
       Only n-1 phases per layer are considered; this is sufficiently general;
       because it defines the unitary apart from a global phase term.
       """
       
       self.U = np.tile(np.eye(self.n),(self.nU,1,1)) #prepares an array containing nU identity matrices, to start with
       
       for ii in range(self.nM-1): 
    
           self.U=self.M[ii,:,:] @ self.U #multiplication (row x column) of the next mixing layer
           self.U[:,0:(self.n-1),:]=self.U[:,0:(self.n-1),:] * np.exp( 1j * self.phi[:, ii*(self.n-1):(ii+1)*(self.n-1)] )[:,:,np.newaxis] 
           #Takes the relevant phase layer phi[:, ii*(n-1):(ii+1)*(n-1)]
           # and adds a new dimension. 
           #In this way the first dimension is the number of
           # the unitary, the second is the row number, the third 
           #is the column number (which here has size 1)
           # Multiplication of U by this array (exploiting broadcasting) 
           # means that each row is multiplied by the phase term. 
           # The first mode is not multiplied by the phase.
           
       #last layer
       self.U = self.M[self.nM-1,:,:] @ self.U
    
            
    def phasesFromCurrAndA(self,I2,A):
        """
        Calculates phases from nU vectors of squared currents I2, 
        given the cross-talk coefficients A. Stores the phases in the object
        as self.phi.
        
        I2 is an array with (nU,n(n-1)) entries, namely containing nU vectors 
        of n(n-1) currents
        A is a 1D array with n*(n-1)^2+n*(n-1) elements; the first n*(n-1)^2 
        are cross-talks, the following n*(n-1) are phases at rest.
        """
                    
        self.nU=I2.shape[0]
       
        #FIRST, reshaping
        I2 = I2.reshape(self.nU, self.n, self.n-1, 1)
        C=A[:(self.n*(self.n-1)**2)].reshape(1, self.n, self.n-1, self.n-1)
        phi0=A[(self.n*(self.n-1)**2):].reshape(1, self.n, self.n-1, 1)
        #NOW
        #I2 is an array (nU,n,n-1,1) --- corresponding to nU sets of n vectors with n-1 values (ordered
        #                           in such a way that each vector is a column)
        #C is an array (1,n,n-1,n-1) --- corresponding to n matrices with (n-1)^2 elements (cross-talks
        #               only considered among phases of the same column)
        #phi0 is an array (1, n,n-1,1) --- corresponding to a set of n vectors with n-1 values
        self.phi = (C @ I2 + phi0).reshape(self.nU,self.nPhases)
        
    def phasesFromCurrents(self,I2):
        """
        Calculates phases from nU vectors of squared currents I2, 
        given the cross-talk coefficients stored in the UPP. Stores the phases in the object
        as self.phi.
        """
        self.phasesFromCurrAndA(I2,self.Astored)
        
    def currentsFromPhases(self):
        """
        Calculates the squared currents (I2) to achieve the
        target optical phases phi stored in the object, compensating for thermal cross-talk
        using the A coefficients stored in the object. (by Francesco Ceccarelli
                                                        and Andrea Crespi)
        """
        
        #Takes the coefficients from the object (self.Astored) and 
        #reshape them into a block-diagonal matrix
        Cmatrix=np.zeros((self.nPhases,self.nPhases))
        for ii in range(self.n):
                Cmatrix[ ii*(self.n-1):(ii+1)*(self.n-1), 
                        ii*(self.n-1):(ii+1)*(self.n-1) ] = self.Astored[ 
                    ii * (self.n-1)**2 : (ii+1) * (self.n-1)**2].reshape(
                        self.n-1,self.n-1)
        phi0=self.Astored[(self.n*(self.n-1)**2):]
        
        #Remove phases/heaters that are not actuated.
        #active_indices = np.where(~np.all(Cmatrix == 0, axis=0))[0]
        Cmatrix = Cmatrix[self.Pmask][:, self.Pmask]
        phi0 = phi0[self.Pmask]
        
        
        I2 = np.zeros_like(self.phi)
        
        for ii in range(self.phi.shape[0]): #self.phi may include more phase vectors
            
            phi = self.phi[ii,self.Pmask]
        
            #Calculates squared currents by numerical inversion.
            J = np.linalg.solve(Cmatrix, phi - phi0)
            while np.any(J < 0):
                phi[J < 0] += 2 * np.pi
                J = np.linalg.solve(Cmatrix, phi - phi0)
                
            # Reconstructs final full vectors
            I2[ii,self.Pmask] = J
        
        return I2

    def setThesePhases(self,phi_full):
        """
        Set the internal phase stored in the object as phi_full
        """
        if (phi_full.ndim > 2) or (phi_full.ndim < 1):
            print(f"Provided phase vector has {phi_full.ndim} dimensions.")
            print("This is not accepted.")
            return
        elif phi_full.ndim == 1:
            phi_full=phi_full[np.newaxis,:]
            
        if phi_full.shape[1] == self.nPhases:
            self.nU=phi_full.shape[0]
            self.phi=phi_full
        else:
            print("Provided phase vector has bad size. Not taken.")
        
        return
        
   
    def phasesFromU(self,Utarget,maxIter=100000,tarInf=1e-10,plotGraph=False):
        """
        Given a unitary matrix Utarget, finds the phases that implement it
        with the UPP. Actually, to find the matrix more phases are used internally.
        Then, phases that are not used in the processor are simplified.
        The final matrix adheres to the target one only up to input/output
        phase terms.
        """
        #maxIter is the maximum number of iterations
        #tarInf is the target infidelity
        
        print("Iterative phases otpimization to find the target unitary.")
        nV=self.nM+1 #number of mixing matrices + the inverse of the target
        V=np.zeros((nV,self.n,self.n)).astype('complex') #array of matrices
        V[:-1,:,:]=self.M #the V matrices are the mixing matrices of the UPP
        V[-1,:,:] = Utarget.conj().T #except from the last one which is the inverse
                                    #of the target

        phaseTerms=np.exp(1j*np.random.rand(self.n+2,self.n)*2*np.pi) 
                    #phase terms are introduced before each V matrix in this
                    #decomposition; they are here initialized to random values
        
        fidelityMonitor=np.zeros(maxIter)
        
        start_time=time.time()
        for jj in range(maxIter):
          
            B=np.eye(self.n).astype('complex') #buffer matrix
            
            for ii in range(nV): #calculates the matrix product
                B = B  * (phaseTerms[ii, :])[:,np.newaxis]  #first phases (multiply rows )
                B = V[ii,:,:] @ B # then mixing matrix
            
            #checks fidelity
            fidelityMonitor[jj]=np.abs(np.trace( B )/self.n)**2
            #if it is high enough it stops cycling
            if (1-fidelityMonitor[jj])<tarInf:
                fidelityMonitor=fidelityMonitor[0:(jj-1)]
                break
            
            if jj % 1000 == 0:
                print("Iteration ",jj)
            
            #if not, it performs another update cycle of all matrices
            for ii in range(nV):
                #expose phases
                B = B * phaseTerms[ii, :].conj() #multiply columns, as right multiplication
                                            # of a diagonal matrix
                #update phase list
                theDiagonal= np.diag(B).conj()
                phaseTerms[ii,:] = theDiagonal / np.abs( theDiagonal ) 
                #applies new phases
                B = B * (phaseTerms[ii, :])[:,np.newaxis]  #multiply rows 
                #removes next mixing layer
                B =  V[ii,:,:] @ B @ V[ii,:,:].conj().T
        #end of optimization      
        print("---Elapsed time: ", (time.time() - start_time), " seconds.")
        
        #brings phases back into the UPP; from now on phaseTerms contains
        #phases as real numbers
        phaseTerms=np.angle(phaseTerms)
        phaseTerms=phaseTerms-phaseTerms[:,-1,np.newaxis] #subtracts the phase of the last mode
        phaseTerms=phaseTerms[1:-1,:] #remove external phases
        phaseTerms[0,::2]=phaseTerms[0,::2]-phaseTerms[0,1::2] #first layer only even modes 
        phaseTerms[0,1::2]=0
        phaseTerms[-1,1:self.n-1:2]=phaseTerms[-1,1:self.n-1:2]-phaseTerms[-1,2:self.n-1:2] #last layer only odd modes
        phaseTerms[-1,:self.n-1:2]=0
        phaseTerms=phaseTerms[:,:-1] #removes last mode

        self.nU=1        
        self.phi=phaseTerms.reshape(1,phaseTerms.size) #reshapes and re-import into the processor
        self.calcU() #updates U
        
        if plotGraph==True:
            plt.figure()
            plt.clf()
            plt.semilogy(1-fidelityMonitor)
            plt.xlabel("Iteration number")
            plt.ylabel("Infidelity")
            
    def calibrateUPP(self,setOfUAmpl,I2,A0):
        """
        Returns an optimized vector of coefficients A (which includes
        the cross-talk coefficients and the phases at rest), given:
            setOfUAmpl: a stack of real matrices namely an array (nU,n,n)
            containing only the moduli of nU unitary matrices
            I2: an array (nU,n(n-1)) namely containing nU vectors 
            of n(n-1) currents -- includes the ones not corresponding to used
            phase shifters
            A0: an initial guess for A
        """
        Abuffer=A0.copy()*0 #prepares an empty array which is needed for the function tryAInfidelity
        
        def tryAInfidelity(Atry):
            #function that tries a given set Atry coeff. with a given set of current I2
            #then, it gives the fidelity with respect to setOfUAmpl
            #I2 has the full number of components n*(n-1)
            #Atry has only the number of components nTotCoeff
            nonlocal I2
            nonlocal setOfUAmpl
            nonlocal Abuffer
            nonlocal self
            
            Abuffer[self.Amask]=Atry
            self.phasesFromCurrAndA(I2,Abuffer)
            self.calcU()
            
            return (1-self.amplitudeFidelityWith(setOfUAmpl))

        def callback(xk): #function that prints the current infidelity in the iterations
            nonlocal ii
            ii=ii+1
            print("- ", ii," - Now infidelity", tryAInfidelity(xk))

        
        start_time=time.time()
        #Performs the optimization
        ii=0
        FitRes=sp.optimize.minimize(tryAInfidelity,A0[self.Amask],options={'disp': True},callback=callback)
        
        Afit=A0
        Afit[self.Amask]=FitRes.x
        print("---Elapsed time: ", (time.time() - start_time), " seconds.")
        
        return Afit


###############################################################################
#                       OTHER USEFUL FUNCTIONS                                #
###############################################################################
    
def importEphosCalibration(UPPobject: Mesh, fileName: str):
    """
    Import calibration data from pickle file, formatted according to
    Ciro's conventions.
    """
    with open(fileName, 'rb') as f:
        calibration = pickle.load(f)
        phi0 = calibration['static_phases']   # rad
        Cmatrix = calibration['xt_matrix']       # rad/mA^2
        M = calibration['n_heaters']
    
    if M != UPPobject.nPhases:
        print("Wrong number of phases. Import aborted.")
        return
    
    for ii in range(UPPobject.n):
        UPPobject.Astored[ii * (UPPobject.n-1)**2 : (ii+1) * (UPPobject.n-1)**2
                          ] = Cmatrix[ ii*(UPPobject.n-1):(ii+1)*(UPPobject.n-1), 
                                      ii*(UPPobject.n-1):(ii+1)*(UPPobject.n-1) ].reshape(-1)
                                      
    UPPobject.Astored[(UPPobject.n*(UPPobject.n-1)**2):]=phi0