# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 10:42:38 2025

@author: caime
"""

import pyvisa
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic


#%% Initialize the VISA resource manager
'''
rm = pyvisa.ResourceManager()
resource_name = rm.list_resources()[2]
print(resource_name)
'''
class oscilloscope():
    def __init__(self, resource_name):
        
        self.rm = pyvisa.ResourceManager()
        self.scope = self.rm.open_resource(resource_name) # Connect to the oscilloscope

    def verify_connection(self):
        print(self.scope.query('*IDN?'))             # Query the device's identification to see if it works
        
    def close(self):
        self.scope.close()
                                                             # setting manually the parameters on the oscilloscope
                                                             #to do it is possible to embed the scope in as a class and pass it to the function so it can work as a wrapper

    def measure_multiple_ch(self):                          #take as input the oscilloscope pyvisa class 

        channels = ["CH1", "CH2","CH3"]                                   #it can be generalized to N elements but i only need 2 for now 
        waveforms = {}

        for ch in channels:                        
            self.scope.write(f"DATa:SOUrce {ch}")                        #choosing the channel to query
            ymult = float(self.scope.query("WFMInpre:YMUlt?"))           #asking for the multiplicator
            yzero = float(self.scope.query("WFMInpre:YZEro?"))           #asking for the zero of the y axis
            yoff  = float(self.scope.query("WFMInpre:YOFf?"))            #asking the offset in time
            xincr = float(self.scope.query("WFMInpre:XINcr?"))           #asking the delta t in x axis

            ic(ymult, yzero, yoff, xincr)

            raw = self.scope.query_binary_values("CURVe?", datatype='B', container=np.array) #retrieves the binary data of the curve
            volts = (raw - yoff) * ymult + yzero                    #computes the voltage value from the binary data
            time = np.arange(len(volts)) * xincr                    #return x axis in second

            waveforms[ch] = (time, volts)                           #saves the temporal profile in a new variable
              
        return waveforms
    
    
    
    
    
class AFG ():
    def __init__(self, resource_name):
        
        self.rm = pyvisa.ResourceManager()
        self.afg = self.rm.open_resource(resource_name)
    
    def verify_connection(self):
        print(self.scope.query('*IDN?'))             # Query the device's identification to see if it work
    
    def set_on(self):
        self.afg.write('OUTPut1:STATe ON')
        
    def set_off(self):
        self.afg.write('OUTPut1:STATe OFF')
    
    def close(self):
         self.afg.close()
           
        
#%%test codice

if __name__ == "__main__":
    
    rm= pyvisa.ResourceManager()
    resname = rm.list_resources()
    name1 = resname[2]
    name2 = resname[3]
    afg = AFG(name2) 
    
    oscill = oscilloscope(name1)
    waveforms = oscill.measure_multiple_ch()
    
    plt.plot(waveforms["CH1"][0],waveforms["CH1"][1], label = "ch1")
    plt.plot(waveforms["CH2"][0],waveforms["CH2"][1], label = "ch2")
    #plt.xlim(0, 0.5)
    #plt.ylim(0,3)
    plt.legend()
    plt.show()

#oscill.closeconnection(