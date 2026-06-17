# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 13:57:48 2024

@author: ICELAB_VACUUM
"""

import numpy as np
import matplotlib.pyplot as plt
import time

from uppscale_cnr.calibrator_base_classes import InputSwitcher, OutputMeasurement, CurrentDriver

from upplab.uppinstruments import Qontrol
from upplab.uppinstruments import ThorlabsPM
from upplab.uppinstruments import LFiberSwitch
from upplab.uppinstruments import OPM150

class input_switch(InputSwitcher):
    def __init__(self, switch : LFiberSwitch.LFiberSwitch, transient = 0.0):
        self.switch : LFiberSwitch.LFiberSwitch = switch
        self.transient = transient
        self.port_memo = self.switch.current_port()
    
    @property
    def port(self):
        return self.port_memo
    
    @port.setter
    def port(self, p):
        current_port = self.port_memo
        
        if p == current_port:
            return
        
        distance_between_channels = np.abs(self.switch.channels[current_port] - \
                                           self.switch.channels[p])
        self.switch.set_port(p)
        self.port_memo = p
        time.sleep(self.transient * distance_between_channels)
    
    def close(self):
        self.switch.close()


class current_driver(CurrentDriver):
    def __init__(self, qnt : Qontrol.Qontrol, transient = 0.0): # Transient is optional, this wrapper can be customized however you like
        self.qnt : Qontrol.Qontrol = qnt
        self.qnt.transient = 0.0
        self.transient = transient
        
        self.current_memo =  np.zeros(len(self.qnt.channels))
    
    @property
    def currents(self):
        return self.qnt.current
    
    @currents.setter
    def currents(self, current_array):
        if len(current_array) != len(self.current_memo):
            raise ValueError("[QTRLWRAP] Lists must be of equal length.")
        
        to_update = [i for i in range(len(current_array))
                     if current_array[i] != self.current_memo[i]]
        
        for idx in to_update:
            self.qnt.set_current(idx, current_array[idx])
        
        self.current_memo = current_array.copy()
        time.sleep(self.transient)
    
    def set_all_zero(self):
        self.currents = np.zeros(len(self.qnt.channels))
    
    def close(self):
        self.qnt.close()

class output_measurement_no_switch(OutputMeasurement):
    def __init__(self, pm : ThorlabsPM.ThorlabsPM, switch, pmn = None, N = 6, transient = 0.0):
        self.pm : ThorlabsPM.ThorlabsPM = pm
        self.switch = switch
        self.pmn = pmn
        self.N = N
        self.transient = transient
    
    def read(self):
        to_return = []
        
        if self.pmn is None:
            read_func = lambda : self.pm.read()
        else:
            read_func = lambda : self.pm.read() / self.pmn.read()
        
        for i in range(self.N):
            # self.switch.set_port(i)
            time.sleep(self.transient)
            to_return.append(read_func())
            
        return to_return
    
    def close(self):
        self.pm.close()

class output_measurement(OutputMeasurement):
    def __init__(self, pm : ThorlabsPM.ThorlabsPM, switch, pmn = None, N = 6, transient = 0.0):
        self.pm : ThorlabsPM.ThorlabsPM = pm
        self.switch = switch
        self.pmn = pmn
        self.N = N
        self.transient = transient
    
    def read(self):
        to_return = []
        
        if self.pmn is None:
            read_func = lambda : self.pm.read()
        else:
            read_func = lambda : self.pm.read() / self.pmn.read()
        
        for i in range(self.N):
            self.switch.set_port(i)
            time.sleep(self.transient)
            to_return.append(read_func())
            
        return to_return
    
    def close(self):
        self.pm.close()



class output_measurement_santec(OutputMeasurement):
    def __init__(self, pm : OPM150.OPM150, N = 12, transient = 0.0):
        self.pm : OPM150.OPM150 = pm
        self.N = N
        self.channels = list(range(N))
        self.transient = transient
    
    def read(self):
        all_channels_dBm = np.array(self.pm.read_multiple_channels())
        all_channels_power = 10**(all_channels_dBm / 10)
        to_return = all_channels_power[self.channels]
        time.sleep(self.transient)        
        return to_return
    
    def close(self):
        self.pm.close()