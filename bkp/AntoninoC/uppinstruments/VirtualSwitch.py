# -*- coding: utf-8 -*-
"""
Created on Wed May 24 14:49:47 2023

@author: niki
"""

from upplab.upp_tools import UPPSwitch
    
class VirtualSwitch(UPPSwitch):
    
    def __init__(self, serial_number = '#', channels = [0]):
        
        self.serial_number = serial_number
        self.channels = channels
        self.current_port = 0
        self.home = 0
        print(f"[VirtualSwitch] VS successfully initialized with serial {self.serial_number}.")
    
    def set_port(self, port):
        try:
            print(f"[VirtualSwitch] Setting port {port} (channel {self.channels[port]}).")
            self.current_port = self.channels[port]
        except:
            print(f"[VirtualSwitch] ERROR: Port {port} has no associated channel.")
    
    def get_port(self):
        return self.current_port
    
    def go_home(self):
        self.set_port(self.home)
    
    def close(self):
        print(f"[VirtualSwitch] Closed connection with VS with serial {self.serial_number}.")