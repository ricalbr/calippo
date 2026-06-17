# -*- coding: utf-8 -*-
"""
Created on Thu May 18 15:29:35 2023

@author: niki
"""

from upplab.upp_tools import UPPReadout
import numpy as np
    
class VirtualPowerMeter(UPPReadout):
    
    def __init__(self, serial_number = '#'):
        
        self.serial_number = serial_number
        self.value = 0.0
        print(f"[VirtualPowerMeter] VPM successfully initialized with serial {self.serial_number}.")
        
    def read(self):
        return self.value
    
    def close(self):
        print(f"[VirtualPowerMeter] Closed connection with VPM with serial {self.serial_number}.")


class VirtualPowerMeterRandom(UPPReadout):
    
    def __init__(self, serial_number = '#', amplitude = 1.0):
        
        self.serial_number = serial_number
        self.amplitude = amplitude
        print(f"[VirtualPowerMeter] VPM successfully initialized with serial {self.serial_number}.")
        
    def read(self):
        return np.random.uniform(0.0, self.amplitude)
    
    def close(self):
        print(f"[VirtualPowerMeter] Closed connection with VPM with serial {self.serial_number}.")


class VirtualPowerMeterSin(UPPReadout):
    
    def __init__(self, serial_number = '#', amplitude = 1.0, frequency = 1.0, phase = 0.0):
        
        self.serial_number = serial_number
        self.amplitude = amplitude
        self.freq = frequency
        self.phase = phase
        self.tick = 0
        print(f"[VirtualPowerMeter] VPM successfully initialized with serial {self.serial_number}.")
        
    def read(self):
        to_return = 0.5*self.amplitude*(1 + np.sin(self.freq*self.tick))
        self.tick += 1
        return to_return
    
    def randomize_phase(self):
        self.phase = np.random.uniform(0.0, np.pi)
    
    def close(self):
        print(f"[VirtualPowerMeter] Closed connection with VPM with serial {self.serial_number}.")

class VirtualDoublePowerMeter(UPPReadout):
    
    def __init__(self, serial_number_1 = '#', serial_number_2 = '#', amplitude = 1.0):
        
        self.serial_number_1 = serial_number_1
        self.serial_number_2 = serial_number_2
        self.amplitude = amplitude
        print(f"[VirtualPowerMeter] Double VPM successfully initialized with serials {self.serial_number_1}, {self.serial_number_2}.")
        
    def read(self):
        return np.random.uniform(0.0, self.amplitude, 2)
    
    def close(self):
        print(f"[VirtualPowerMeter] Closed connection with VPM with serial {self.serial_number_1}.")
        print(f"[VirtualPowerMeter] Closed connection with VPM with serial {self.serial_number_1}.")