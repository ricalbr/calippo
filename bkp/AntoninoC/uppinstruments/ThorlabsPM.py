# -*- coding: utf-8 -*-
"""
Created on Thu May 18 15:21:06 2023

@author: niki
"""

from upplab.upp_tools import UPPReadout
from ThorlabsPM100.ThorlabsPM100 import ThorlabsPM100
import matplotlib.pyplot as plt
import pyvisa as visa
import numpy as np
import time

class ThorlabsPM(UPPReadout):
    """ Thorlabs Power Meter object.
    Args:
        serial (str): The serial number of the testina you want to connect.
        
        wavelength (int or float, optional): The wavelength that will be set once the
            connection is established.
        
        
    Raises:
        AssertionError: If the connection fails raises an exception.
        
        ValueError: if the wavelength to be set is not supported by the 'Testina'
    Examples:
        TBD
    Note:
        Remember to delete the object when it is not needed anymore, otherwise the
        USB link will remain busy:
        >>> power_meter.close()
    """
    
    def __init__(self, serial, wavelength = None):
        
        self.serial = serial

        self.rm = visa.ResourceManager()
        rs = self.rm.list_resources()
        
        if not any([self.serial in resource for resource in rs]):
            raise ValueError(f'No device was found with serial {self.serial}')
        
        # Find resource with serial string inside and get its full USB address
        self.USB_address = list(filter(lambda x: self.serial in x, rs))[0]
        
        # Get instance of resource
        self.inst = self.rm.open_resource(self.USB_address)
        
        # This seems to fix the Timeout error, not sure why
        self.inst.clear()
        
        # Create ThorlabsPM100 object
        self.pm = ThorlabsPM100(inst = self.inst)
        
        if wavelength:
            self.wavelength = wavelength
        
        # Verbose mode to print many things
        self.verbose = False
        
        # Averaging over several measurements with timeout
        self.average_count = 1
        self.average_time = 0.1
    
    @property
    def maximum_wavelength(self):
        return self.pm.sense.correction.maximum_wavelength
    
    @property
    def minimum_wavelength(self):
        return self.pm.sense.correction.minimum_wavelength
    
    @property
    def wavelength(self):
        return self.pm.sense.correction.wavelength
    
    @wavelength.setter
    def wavelength(self, wl):
        if self.minimum_wavelength <= wl <= self.maximum_wavelength:
            self.pm.sense.correction.wavelength = wl
        else: raise ValueError(f'Wavelength {wl} nm not supported by power meter')
    
    def read(self):
        """ Returns the power measured in Watts (float) """
        measurement = [self.pm.read]
        if self.average_count > 1:
            for i in range(self.average_count):
                time.sleep(self.average_time)
                measurement.append(self.pm.read)
        
        to_return = np.mean(measurement)
        if self.verbose:
            print(f'[ThorlabsPM] read: {to_return*1000:.3f} mW')
        return to_return
    
    def continuous_read(self, dt = 0.1):
        '''
        Continuously read power every dt seconds

        Parameters
        ----------
        dt : float, optional
            amount of time in seconds between reads. The default is 0.1.

        Returns
        -------
        None.

        '''
        while True:
            print(f'{self.read()*1000:.3f} mW')
            time.sleep(dt)
    
    def continuous_graph(self, dt = 0.1):
        '''
        Allows for continuous measurement of power with a given time interval,
        useful for manual alignment since TL power meters don't have a screen.
    
        Parameters
        ----------
        dt : int or float, optional
            Time interval for continuous measurement. The default is 0.1 seconds.
    
        Returns
        -------
        None.
    
        '''
    
        t_vals = [0]
        p_vals = [self.read()*1000]
        plt.plot(t_vals, p_vals)
        plt.ylabel('Optical Power (mW)')
        
        while True:
            t_vals.append(t_vals[-1]+1)
            p_vals.append(self.read()*1000)
            
            plt.gca().lines[0].set_xdata(t_vals)
            plt.gca().lines[0].set_ydata(p_vals)
            plt.gca().relim()
            plt.gca().autoscale_view()
            plt.pause(0.05)
            time.sleep(dt)
        
        plt.show()
    
    def __str__(self):
        """ Prints status of power meter """
        if self.pm:
            string_to_print = f'TL Power Meter with serial {self.serial} operating at wavelength {self.wavelength}'
        else: string_to_print = f'TL Power Meter with serial {self.serial}, communication has been closed'
        return string_to_print
        
    def close(self):
        if self.pm: 
            self.pm = None
            self.inst.close()
            self.rm.close()