# -*- coding: utf-8 -*-
"""
Created on Thu May 18 14:08:57 2023

@author: niki
"""

from upplab.upp_tools import UPPCurrentSource

class VirtualCurrentSource(UPPCurrentSource):
    
    def __init__(self, init_mode = 'i'):
        
        if init_mode.lower() not in ['v','i']:
            raise AssertionError("[VirtualCurrentSource] Error: qontrol manager must be initialised with either voltage (v) or current (i) mode.")
        else:
            self._init_mode = init_mode.lower()
            print(f"[VirtualCurrentSource] VCS manager successfully initialised in {self._init_mode} mode.")
        
        self.channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.verbose = True
        
    def set_current(self, shifter_index, current):
        assert self._init_mode == 'i', "Error: Trying to set current in voltage mode."
        
        ch = self.channels[shifter_index]
        if self.verbose:
            print(f"[VirtualCurrentSource] Channel {ch} set to {current} mA. :)")
    
    def get_current(self, shifter_index):
        return
    
    def set_all_zero(self):
        return
    
    def close(self):
        return
    
        
        