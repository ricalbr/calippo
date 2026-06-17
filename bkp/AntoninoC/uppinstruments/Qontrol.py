# -*- coding: utf-8 -*-
"""
Created on Thu May 18 14:06:51 2023

@author: niki
"""

from upplab.upp_tools import UPPCurrentSource
import qontrol
import time
import numpy as np

class Qontrol(UPPCurrentSource):
    
    def __init__(self, serial_name, init_mode='i', imax = 24, vmax = 12, verbose = False):
        """Initializer

        It sets up the necessary

        Parameters
        ----------
        serial_name : string
            Name of the serial port the Qontrol board is physically connected to (e.g. 'COM3').
        init_mode : char, optional
            Initialization mode, either "voltage" ('v') or "current" ('i'); the
            initialization mode determines which of the two functions (set_voltage
            or set_current) can be used. The default is 'v'.
        imax: float, optional
            value of the compliance current (in mA), default is 24 mA
        vmax: float, optional
            value of the compliance voltage (in V), default is 12 V

        """
        
        self._q = qontrol.QXOutput(serial_port_name = serial_name, response_timeout = 0.1)
        self.set_compliance(current = imax, voltage = vmax)
        
        self.imax = imax
        self.vmax = vmax
        
        if init_mode.lower() not in ['v','i']:
            raise AssertionError("[QTRL] Error: qontrol manager must be initialised with either voltage (v) or current (i) mode.")
        else:
            self._init_mode = init_mode.lower()
            print(f"[QTRL] Qontrol manager successfully initialised in {self._init_mode} mode.")
        
        self.channels = []
        self.transient = 0.5
        self.verbose = verbose

    @property
    def current(self):
        return self._q.i[:] # returns a list of float
    
    @current.setter
    def current(self, current_array):
        
        to_update = [i for i in range(len(current_array))]
        
        for idx in to_update:
            self.set_current(idx, current_array[idx])
        
        time.sleep(self.transient)
    
    def set_current(self, shifter_index, current):
        """Current setter

        Parameters
        ----------
        shifter_index : int
            Index of shifter of interest. The current will be set to "value" for the channel
            corresponding to the shifter only.
        current : float
            Current value to be set, in milliAmperes.

        NOTE: Exclusively works if current mode ('i') has been selected when initializing.
        """
        assert self._init_mode == 'i', "Error: Trying to set current in voltage mode."
        
        ch = self.channels[shifter_index]
        if np.greater(current, self.imax):
            print(f"[QTRL] Amp value too high, setting to 0 mA on Qontrol channel {ch}.")
            self._q.i[ch] = 0
        else:
            self._q.i[ch] = float(current)
        
        if self.verbose:
            print(f"[QTRL, v] Setting channel {ch} (Resistance {shifter_index}) to {current} mA.")
        
        time.sleep(self.transient)
    
    def get_current(self, shifter_index):
        """Current getter

        Parameters
        ----------
        shifter_index : int
            Index of shifter of interest. 
        """
        ch = self.channels[shifter_index]
        return self._q.i[ch]
    
    @property
    def voltage(self):
        return self._q.v[:] # returns a list of float
    
    
    def set_voltage(self, shifter_index, voltage):
        """Voltage setter

        Parameters
        ----------
        ch : int
            Index of shifter of interest. The voltage will be set to "value" for the channel
            corresponding to the shifter only.
        voltage : float
            Voltage value to be set, in Volts.
        transient : float
            Temperature transient to wait after setting current, default is 0.5 seconds.

        NOTE: Exclusively works if voltage mode ('v') has been selected when initializing.
        """
        assert self._init_mode == 'v', "Error: Trying to set voltage in current mode."
        
        ch = self.channels[shifter_index]
        if np.greater(voltage, self.vmax):
            print(f"[QTRL] Volt value too high, setting to 0 V on Qontrol channel {ch}.")
            self._q.v[ch] = 0
        else:
            self._q.v[ch] = float(voltage)
        
        if self.verbose:
            print(f"[QTRL, v] Setting channel {ch} (Resistance {shifter_index}) to {voltage} V.")
        
        time.sleep(self.transient)
    
    def get_voltage(self, shifter_index):
        """Voltage getter

        Parameters
        ----------
        shifter_index : int
            Index of shifter of interest.
        """
        ch = self.channels[shifter_index]
        return self._q.v[ch]
    
    @property
    def power(self):
        """ Returns list of powers for each channel in mW """
        return [v*i for (v,i) in zip(self.voltage, self.current)] # mW
    
    
    def set_compliance(self, current=24, voltage=12):
        """Set Compliance

        Sets the value of compliance for voltage and current.
        If the Qontrol board measures on some channel a higher value of
        current or voltage it will set the channel to zero. This can happen at
        any time, not necessarily when calling get_voltage or get_current.

        Parameters
        ----------
        current : float, optional
            Current value (in mA) for the compliance. The default is 24 mA.
        voltage : float, optional
            Voltage value (in V) for the compliance. The default is 12 V.
        """
        assert 0 < current <= 24 and 0 < voltage <= 12, AssertionError('[QTRL] set_compliance: Trying to set compliance current/voltage to unacceptable value')
        self._q.imax[:] = current
        self._q.vmax[:] = voltage
        self.imax = current
        self.vmax = voltage

    def set_all_zero(self):
        """Set All Zero

        Sets all channels to zero voltage.

        """
        if self.verbose:
            print('[QTRL] Setting all currents to zero.')
        self._q.v[:] = 0.0

    def close(self):
        """Close

        Closes all streams and communications from the device. It is mandatory
        to call this function once Qontrol actions are no longer needed, such
        as at the end of a script.
        """
        if self._q:
            self.set_all_zero()
            self._q.close()
            self._q = None
            print('[QTRL] Successfully closed communication with all Qontrol channels.')
        else:
            print('[QTRL] Communication with Qontrol device was already closed!')