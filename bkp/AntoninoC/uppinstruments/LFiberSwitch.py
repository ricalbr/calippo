# -*- coding: utf-8 -*-
"""
Created on Fri May 13 11:15:53 2022

@author: ICE
"""

from upplab.upp_tools import UPPSwitch
import pyvisa as visa
import time

class LFiberSwitch(UPPSwitch):
    
    def __init__(self, n_in, n_out, resource= 'ASRL9::INSTR', transient = 0.1, verbose = True):
        '''
        This class represents a fiber switch, in particular an Lfiber (PM) optical switch 1x16.
        Can work for more than 16 output just by changing self.max_output_channel.
        Communicates with the device through serial communication provided by pyvisa 

        Parameters
        ----------
        resource : String representing the resource of the fiber switch connected in the COM port of the PC
            The default is 'ASRL8::INSTR'.
        n_out : INT representing the number of output fibers in the fiber switch 
        n_in  : INT representing the number of input fibers in the fiber switch
        Returns
        -------
        None.

        '''
        #Commands to initialize 
        rm = visa.ResourceManager("@py")
        self.my_instrument = rm.open_resource(resource)
        self.my_instrument.read_termination = '>'
        self.my_instrument.write_termination = '>'
        self.my_instrument.query_delay = 0.1
        
        self.verbose = verbose
        self.transient = transient
        
        self.n_in = n_in
        self.n_out = n_out
        self.channels = []
        
        print('[LFiberFS] Connection with fiber switch successfully initialized.')
        

    def convert_channel(self, channel):
        '''
        given a numerical channel (INT) it converts it into char into a readable way for the fiber switch

        '''
        if len(str(channel))!=1:
            channel = str(channel)
        else:
            channel = '0' + str(channel)
        return channel
            
    def set_port(self, port):
        
        ''' given a channel (INT or STR) tries to set the fiber switch to provide light to that channel.
        prints if succesful or if any error occured
        for a 2xn optical switch sets automatically the first input to the desired channel then the second input to the next channel
        '''
        
        channel = self.channels[port]
        
        if self.n_in == 1:
            output=self.my_instrument.query('<OSW_OUT_' + self.convert_channel(channel))
        elif self.n_in == 2:
            output=self.my_instrument.query('<OSW_OUT_00_' + self.convert_channel(channel))
        else: print('[LFiberFS] Trying to set a fiber switch with more than 2 inputs. Code cannot handle this yet!')
        
        if output== '<OSW_OUT_OK':
            if self.verbose: print(f'[LFiberFS, v] Succesfully set LFFS to port {port} (channel {channel}).')
        elif output=='<OSW_OUT_OVERFLOW':
            print(f'[LFiberFS] Error: cannot set channel to {channel}. Avaiable channels span from 1 to {self.max_output_channel}. Current channel: {self.current_channel()}.')
        else:
            raise AssertionError(f'[LFiberFS] Error: could not switch the channel. Error code:\n{output}')
        
        time.sleep(self.transient)
        
    def current_port(self):
        return self.channels.index(self.current_channel())
    
    def current_channel(self):
        '''Returns the current channel of the fiber switch '''
        output = self.my_instrument.query('<OSW_OUT_?')
        numbers = [i for i in range (17) if str(i) in output]
        print(f'[LFiberFS] Current channel is set to {numbers[1:]}')
        return numbers[1]
    
    def go_home(self):
        '''Resets the fiber switch, light should not come out of any port'''
        if self.n_in == 1:
            output= self.my_instrument.query('<OSW_OUT_00')
        elif self.n_in == 2:
            output= self.my_instrument.query('<OSW_OUT_00_00')  
        if output== '<OSW_OUT_OK':
            print('[LFiberFS] Fiber switch succesfully reset.')
        else:
            raise AssertionError(f'[LFiberFS] Could not reset the switch. Error code:\n{output}')
        
        time.sleep(self.transient)
        
    def switch_type(self):
        '''Returns the informations about the fiber switch'''
        print(self.my_instrument.query('<OSW_TYPE_?'))

    def close (self):
        '''Closes all the communications'''
        self.my_instrument.before_close()
        self.my_instrument.close()
        print('[LFiberFS] Connection with fiber switch successfully closed.')

