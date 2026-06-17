"""
Created on Fri February 02 2024

@author: Vittorio Grimaldi

Some general considerations:
-   Inside the methods, I used the internal variables names consistent with their documentation.
    The methods instead have been renamed in (I believe) a more intuitive way.
-   The documentation is poorly made and some of the functions do not work the way they claim they do.   
-   Some of the functions described in the documentation are not compatible with the device we own. 
    Notably, the power unit cannot be changed. For this reason, I implemented that manually (see _convert_to_linear method).
-   "Bulk mode" stuff is not implemented, since documentation for one crucial function is missing. 
    TODO should try to implement it nonetheless
-   WARNING: changing channel requires some time. If active_channel is modified, it is advised to wait a minimum 
    amount of time before sending another command. To prove this, try removing the time.sleep(0.1) 
    in the set_autorange_all_channels method or in the set_gain_all_channels method. It won't work anymore.
"""

from upplab.uppinstruments import Optotest
from upplab.upp_tools import UPPReadout
from ctypes import c_bool, c_int, byref, c_char_p, c_uint64, c_double, c_byte
import logging
import time
from enum import Enum
from typing import Literal, Optional, Tuple
import sys
import math

class OPM150(UPPReadout):

    # This is black magic in order to obtain function names within the function itself
    def get_func_name(self, n = 0):
        return sys._getframe(n+1).f_code.co_name
        # Original solution: get_func_name = lambda n=0: sys._getframe(n+1).f_code.co_name

    def __init__(self):
        """Initializes the communication with the OPM150 powermeter from Santec."""

        self.logger = logging.getLogger(__name__)
        self._is_connection_open = False

        _dev_count = self.USB_device_count
        self.logger.debug(f"Device count : {_dev_count}")

        # Iterate through all USB devices and find the first OP710
        for i in range(_dev_count):
            _description = self.get_USB_device_description(i)
            self.logger.debug(f"Description : {_description}")
            if "OP710" in _description:
                self.device_number = i
                break

        self.logger.debug(f"Device number : {self.device_number}")
        
        self._handle = self.open_USB_device(self.device_number)
        self.logger.debug(f"USB Handle : {self._handle}")

        self._is_connection_open = self.open_driver(self._handle)
        self.logger.debug(f"Is connection open : {self._is_connection_open}")

        # Initializes internal variables for properties
        # self._backlight = None    # Not supported for OPM-150
        self._remote_mode = None
        # self._power_unit = None   # Not supported for OPM-150
        self._sampling_speed = None # TODO this is for sure channel-related, but I have no way of testing it
        
        self.remote_mode = True
        self.power_unit : Literal[0,1] = 0 # Since it is not supported, I had to do a manual conversion in the read_power method
        
        self.available_wavelengths = Optotest.Wavelengths
        
        # If you close and open the communication again, the getter of the active_channel will return 1
        # if queried. However, the actual active_channel remains the one that was set before closing the communication
        # with the device. To prevent mistakes, I call the setter of the active_channel here in the initializer.
        self.active_channel = 1 
        
    def __del__(self):
        # self.logger.debug("Deleter called")
        self.close()

    def close(self) -> None:
        """Closes the driver, closing the connection and communication with the device."""
        if self._is_connection_open:
            self.remote_mode = False
            _ret = Optotest.CloseDriver()
            self._check(self.get_func_name(), _ret)
            self._is_connection_open = False

    def get_module_USB_handle(self, module: int) -> int:
        """Checks if module is active and returns USB Handle for that module if it is. Returns 0 otherwise."""
        _module = c_int(module)
        _hanlde = c_int()
        _handle = Optotest.ActiveModule(_module)
        return _handle
    
    def open_USB_device(self, dev_number: int) -> int:
        """
        Opens the specified USB device and gets a USB Handle. The device is specified by USB Device Number. 
        The USB Handle is then used with the open_driver method.
        """
        _dev_number = c_int(dev_number)
        _handle = c_uint64()
        _ret = Optotest.OpenUSBDevice(_dev_number, byref(_handle)) 
        self._check(self.get_func_name(), _ret)
        return _handle.value

    def open_driver(self, handle : int = 0) -> bool:
        """
        Initializes the device with the given USB Handle.
        Optionally, 0 can be used as a shortcut to open the first USB device detected.
        Returns True if successful.
        """
        _handle = c_uint64(handle)
        _ret = Optotest.OpenDriver(_handle)
        self._check(self.get_func_name(), _ret)
        if _ret == 0:
            return True
        else:
            return False

    @property
    def USB_device_count(self) -> int:
        """Gets the number of connected compatible USB devices."""
        _count = c_int()
        _ret = Optotest.op_dll.GetUSBDeviceCount(byref(_count))
        self._check(self.get_func_name(), _ret)
        return _count.value

    def get_USB_device_description(self, dev_number: int) -> int:
        """Gets the USB Device Description (up to 16 characters) of the specified device."""
        # Caller must allocate memory for deviceDescription buffer (16 characters + 1 null-terminator)
        _description = c_char_p()
        _dev_number = c_int(dev_number)
        _ret = Optotest.GetUSBDeviceDescription(_dev_number, byref(_description))
        self._check(self.get_func_name(), _ret)
        return _description.value.decode("UTF-8")

    @property
    def USB_serial_number(self) -> int:
        """Gets the USB Device Serial Number (8 characters) of currently selected device."""
        # Caller must allocate memory for deviceDescription buffer (16 characters + 1 null-terminator)
        _serial = c_char_p()
        _device_number = c_int(self.device_number)
        _ret = Optotest.GetUSBSerialNumber(_device_number, byref(_serial))
        self._check(self.get_func_name(), _ret)
        return _serial.value.decode("UTF-8")

    @property
    def USB_status(self) -> bool:
        """Gets the USB error flag of last operation as well as the error code of currently selected device."""
        _have_error = c_bool()
        _device_number = c_int(self.device_number)
        _ret = Optotest.GetUSBStatus(_device_number, byref(_have_error))
        self._check(self.get_func_name(), _ret)
        return _have_error.value
    
    @property
    def dll_revision(self) -> int:
        """Gets the revision number of DLL."""
        _revision = c_int()
        _revision = Optotest.GetDLLRev()
        return _revision

    @property
    def fw_revision(self) -> int:
        """Gets the device's firmware revision."""
        _revision = c_int()
        _revision = Optotest.GetFWRevision()
        return _revision

    @property
    def status(self) -> Optotest.ErrorCodes:
        """Gets the initialization status of the DLL and USB device."""
        _status = c_int()
        _status = Optotest.GetDLLStatus()
        # self._check(_status)
        return Optotest.ErrorCodes(_status)

    @property
    def module_ID(self) -> Optotest.ModuleID:
        """Gets the Module ID (internal product code) of the device."""
        _id = c_int()
        _ret = Optotest.GetModuleID(byref(_id)) # There is a return even if the documentation is not super-clear about it 
        self._check(self.get_func_name(), _ret)
        return Optotest.ModuleID(_id.value)

    @property
    def selected_device(self) -> int:
        """Gets the device number of currently selected module."""
        _number = c_int()
        _ret = Optotest.GetModuleNumber(byref(_number))
        self._check(self.get_func_name(), _ret) # If _ret == -1, it means that the module is not initialized
        return _number.value 

    @selected_device.setter
    def selected_device(self, device: int) -> None:
        """
        Selects the specified device. All subsequent USB commands will be directed to the selected module. 
        The module number corresponds to the device number.
        """
        _device = c_int(device)
        _ret = Optotest.SelectModule(_module)
        self._check(self.get_func_name(), _ret)


    # Not supported for OPM-150
    # @property
    # def backlight(self) -> Optional[bool]:
    #     """
    #     Checks whether the front panel display backlight is on (True) or off (False).
    #     Returns None if the backlight setter was never called.
    #     """
    #     return self._backlight

    # @backlight.setter
    # def backlight(self, on: bool) -> None:
    #     """Turns the front panel display backlight on or off. Doesn't work though."""
    #     self._backlight = on # Saves value in internal variable 
    #     _bl = c_int(1) if on else c_int(0)
    #     _ret = Optotest.Backlight(_bl)
    #     self._check(self.get_func_name(), _ret)
        
    @property
    def active_channel(self) -> int:
        """Gets the current active channel."""
        _ch = c_int()
        _ret = Optotest.GetActiveChannel(byref(_ch))
        self._check(self.get_func_name(), _ret)
        return _ch.value

    @active_channel.setter
    def active_channel(self, ch: int) -> None:
        """Sets the current active channel."""
        _ch = c_int(ch)
        _ret = Optotest.SetActiveChannel(_ch)
        self._check(self.get_func_name(), _ret)

    def read_temperature(self, unit : Literal[0,1,2] = 1) -> int:
        """
        Gets the temperature in the specified units.
        unit:
            0 - raw digital value
            1 - Celsius
            2 - Fahrenheit
        """
        _temperature = c_double() 
        _ret = Optotest.GetTemperature(byref(_temperature), unit)
        self._check(self.get_func_name(), _ret)
        return _temperature.value

    @property
    def wavelength(self) -> Optotest.Wavelengths:
        """Gets the active channel's current wavelength information."""
        _wavelength = c_int()
        _index = c_int()
        _count = c_int() # N.B. it returns 6, but there are actually 8 wavelengths available
        _ret = Optotest.GetWavelength(byref(_wavelength), byref(_index), byref(_count))
        self._check(self.get_func_name(), _ret)
        self.logger.debug(f"Wavelength:{_wavelength.value}")
        self.logger.debug(f"Index wavelength:{_index.value}")
        self.logger.debug(f"Count of available wavelength:{_count.value}")
        return Optotest.Wavelengths(_wavelength.value)

    
    @wavelength.setter
    def wavelength(self, wavelength: int) -> None:
        """
        Sets the active channel's measurement wavelength. 
        
        WARNING: In the documentation, it states that: 
        'This will attempt to find the closest match from the internally defined list of discrete wavelengths'.
        This is not true however, and you have to pass the exact wavelength for it to change. For this reason,
        I preferred to create an enum in the Optotest module.
        """
        try:
            wl = self.available_wavelengths(wavelength)
        except ValueError:
            raise ValueError(f'''Wavelength {wavelength} is not available.\n
                             Available wavelengths are {[w.value for w in self.available_wavelengths]}.''')
        _wavelength = c_int(wl.value)
        _ret = Optotest.SetWavelength(_wavelength)
        self._check(self.get_func_name(), _ret)


    def set_next_wavelength(self) -> Optotest.Wavelengths:
        """
        Sets wavelength to the next one the active channel's list of available measurement wavelength.
        Returns the value of the actual wavelength set. 
        """
        _wavelength = c_int()
        _index = c_int()
        _count = c_int()
        _ret = Optotest.NextWavelength(byref(_wavelength), byref(_index), byref(_count))
        self._check(self.get_func_name(), _ret)
        self.logger.debug(f"Wavelength:{_wavelength.value}")
        self.logger.debug(f"Index wavelength:{_index.value}")
        self.logger.debug(f"Count of available wavelength:{_count.value}")
        return Optotest.Wavelengths(_wavelength.value)

    def read_analog(self) -> int:
        """Reads raw ADC values for current active OPM channel. Returns the analog reading and the gain."""
        _analog = c_int()
        _gain = c_int() 
        _mode = c_int()
        _ret = Optotest.ReadAnalog(byref(_analog), byref(_gain), byref(_mode))
        self.logger.debug(f"Analog value reading:{_analog.value}")
        self.logger.debug(f"Gain:{_gain.value}")
        self.logger.debug(f"Mode:{_mode.value}")
        self._check(self.get_func_name(), _ret)
        return _analog.value

    def convert_analog_reading(self, analog: int, gain: int) -> float:
        """
        Converts a pair of raw ADC value and gain (from read_analog method) to power.
        If power_unit == 0, the result is in [dBm], if power_unit == 1, the result is in [W].
        """
        _analog = c_int(analog)
        _gain = c_int(gain)
        _power = c_double()
        _ret = Optotest.ConvertPower(_analog, _gain, byref(_power))
        self._check(self.get_func_name(), _ret)
        if self.power_unit == 0:
            return _power.value
        elif self.power_unit == 1:
            return self._convert_to_linear(_power.value)
        else:
            raise ValueError("power_unit should only be 0 or 1.")


    def read_power(self) -> float:
        """
        Reads power from active channel. 
        If power_unit == 0, the result is in [dBm], if power_unit == 1, the result is in [W].
        """
        _power = c_double()
        _ret = Optotest.ReadPower(byref(_power))
        self._check(self.get_func_name(), _ret)
        if self.power_unit == 0:
            return _power.value
        elif self.power_unit == 1:
            return self._convert_to_linear(_power.value)
        else:
            raise ValueError("power_unit should only be 0 or 1.")
    
    def read_multiple_channels(self, channels: list[int] = [i+1 for i in range(24)], sleep: float = 0.1) -> list[float]:
        """
        Reads power from a list of channels and returns a list of powers.
        WARNING: channels go from 1 to 24! Not from 0!
        WARNING: the channels are queried in the order given! So if channels = [1, 7, 3], then powers is a list
        containing the power reading of channel 1, 7 and 3 in this order.
        """
        self.update_channels_buffer()
        time.sleep(sleep) # I wouldn't remove this, but I'm not sure
        powers = list()
        for ch in channels:
            powers.append(self.read_channel_buffer_power(ch))
            # time.sleep(sleep) # Unclear if it is needed here
        return powers


    def update_channels_buffer(self) -> None:
        """Measure power from all channels simultaneously and update internal buffers."""
        _ret = Optotest.GetChannelBuffer()
        self._check(self.get_func_name(), _ret)


    def read_channel_buffer_power(self, ch: int) -> float:
        """
        Reads the power measured by update_channels_buffer method for the specified channel. 
        Must call the update_channels_buffer first.
        """
        _ch = c_int(ch)
        _power = c_double()
        _ret = Optotest.ReadChannelBuffer(_ch, byref(_power))
        self._check(self.get_func_name(), _ret)
        if self.power_unit == 0:
            return _power.value
        elif self.power_unit == 1:
            return self._convert_to_linear(_power.value)
        else:
            raise ValueError("power_unit should only be 0 or 1.")
    
    def read_channel_buffer_analog(self, ch: int) -> float:
        """
        Reads the raw ADC power measured by update_channels_buffer method for the specified channel. 
        Must call update_channels_buffer method first.
        """
        _ch = c_int(ch)
        _analog = c_uint16() # Not an error, here is c_uint16 as per documentation
        _gain = c_byte() # Not an error, here is a c_byte as per documentation
        _ret = Optotest.ReadChannelBufferRaw(_ch, byref(_analog), byref(_gain))
        self._check(self.get_func_name(), _ret)
        return _analog.value

    def read_relative_power(self) -> float:
        """
        Reads relative power from active channel.
        Must use set_reference_power method to perform a reference beforehand.
        If power_unit == 0, the result is in [dB], if power_unit == 1, the result is adimensional 
        (it's the ratio between the absolute power and the reference).
        """
        _loss = c_double()
        _ret = Optotest.ReadLoss(byref(_loss))
        self._check(self.get_func_name(), _ret)
        if self.power_unit == 0:
            return _loss.value
        elif self.power_unit == 1:
            # This returns the ratio between the absolute power and the reference previously set.
            return self._convert_to_linear_relative(_loss.value) 
        else:
            raise ValueError("power_unit should only be 0 or 1.")

    @property
    def remote_mode(self) -> Optional[bool]:
        """Returns wheter Remote Mode is active or not. Returns None if the setter has never been called."""
        return self._remote_mode

    @remote_mode.setter
    def remote_mode(self, mode: bool = True) -> None:
        """Sets Remote Mode. This will display a static logo and locks out user interaction from the front-panel."""
        self._remote_mode = mode
        _mode = c_int(1) if mode else c_int(0) # Has to be a c_int as per the documentation
        _ret = Optotest.RemoteMode(_mode)
        self._check(self.get_func_name(), _ret)

    # A bit more difficult to implement as a property since there is an autorange for each channel 
    # and no GetAutorange method.
    # @property
    # def autorange(self) -> Optional[bool]:
    #     """Checks whether the autorange is on or off. Returns None if the setter has never been called."""
    #     return self._autorange

    def set_autorange_active_channel(self, autorange: bool) -> None:
        """
        Sets the Auto-Range on or off (Range-Hold) on currently-active channel.
        WARNING: the autorange property is actually shared between adjacent channels.
        """
        _range = c_int(1) if autorange else c_int(0)
        _ret = Optotest.SetAutoRange(_range)
        self._check(self.get_func_name(), _ret)
      
    def set_autorange_all_channels(self, autorange: bool) -> None:
        """Sets the Auto-Range on or off (Range-Hold) for all channels."""
        _current_active_channel = self.active_channel # Save current active channel
        # The autorange property is actually shared between adjacent channels.
        # So if I want to change the autorange for the whole instrument, I have to cycle through
        # all the odd (or even) channels and change the autorange for each of them.
        for i in range(1,25):
            if i%2 == 1:
                print(i)
                self.active_channel = i
                time.sleep(0.1) # It won't work correctly without this sleep 
                self.set_autorange_active_channel(autorange)
        self.active_channel = _current_active_channel # Returns to last active channel used

    # def set_bulk_mode(self) -> None:
    #     pass #documentation missing #TODO should try it nonetheless

    @property
    def gain(self) -> Literal[0,1,2,3,4,5,6,7]:
        """Returns the gain of the currently-active channel."""
        _analog = c_int()
        _gain = c_int() 
        _mode = c_int()
        _ret = Optotest.ReadAnalog(byref(_analog), byref(_gain), byref(_mode))
        self.logger.debug(f"Analog value reading:{_analog.value}")
        self.logger.debug(f"Gain:{_gain.value}")
        self.logger.debug(f"Mode:{_mode.value}")
        self._check(self.get_func_name(), _ret)
        return _gain.value

    @gain.setter
    def gain(self, gain: Literal[0,1,2,3,4,5,6,7]) -> None:
        """
        Sets the gain to the currently-active channel. This will automatically disable the autorange.
        WARNING: the gain property is actually shared between adjacent channels.
        """
        _gain = c_int(gain)
        _ret = Optotest.SetGain(_gain)
        self._check(self.get_func_name(), _ret) 

    def set_gain_all_channels(self, gain: Literal[0,1,2,3,4,5,6,7]) -> None:
        """Sets the gain to all channels. This will automatically disable the autorange."""
        _current_active_channel = self.active_channel # Save current active channel
        # The gain property is actually shared between adjacent channels.
        # So if I want to change the gain for the whole instrument, I have to cycle through
        # all the odd (or even) channels and change the gain for each of them.
        for i in range(1,25):
            if i%2 == 1:
                self.active_channel = i
                time.sleep(0.1) # It won't work correctly without this sleep 
                self.set_gain_active_channel(gain)
        self.active_channel = _current_active_channel # Returns to last active channel used


    # Not supported for OPM-150       
    # @property
    # def power_unit(self) -> Optional[Literal[0,1]]:
    #     """Checks whether the power is measured in [dBm] (0) or [W] (1). Returns None if the setter has never been called."""
    #     return self._power_unit
        
    # @power_unit.setter
    # def power_unit(self, unit: Literal[0,1]) -> None:
    #     """Sets power measurement mode. 0 for [dBm], 1 for [W]."""
    #     self._power_unit = unit 
    #     _unit =  c_int(unit)
    #     _ret = Optotest.SetOPMMode(unit)
    #     self._check(self.get_func_name(), _ret)

    @property
    def sampling_speed(self) -> Optional[Literal[0,1,2,3,4,5,6,7,8]]:
        """Returns the sampling speed selected for the device. Returns None is the setter has never been called."""
        return self._sampling_speed

    @sampling_speed.setter
    def sampling_speed(self, speed: Literal[0,1,2,3,4,5,6,7,8]) -> None:
        """
        Sets the OPM Sampling Speed. Note that faster sampling speeds can result in less stable measurements.
        speed goes from 0-8, 0 is fastest, 9 is slowest; each increment doubles the rate of the previous.
        # TODO how can i test this?
        # TODO probably it's a property of the single channels, but I have no way of testing it
        """
        self._sampling_speed = speed
        _speed = c_byte(speed)
        _ret = Optotest.SetSamplingSpeed(_speed)
        self._check(self.get_func_name(), _ret)

    def get_reference_power(self) -> float:
        """
        Returns the internal power reference for active OPM channel.
        To change it to the current power, use the set_reference_power method. 
        If power_unit == 0, the result is in [dBm], if power_unit == 1, the result is in [W].
        """
        _ref_power = c_double()
        _ret = Optotest.ReferencePower(byref(_ref_power))
        self._check(self.get_func_name(), _ret)
        if self.power_unit == 0:
            return _ref_power.value
        elif self.power_unit == 1:
            return self._convert_to_linear(_ref_power.value)
        else:
            raise ValueError("power_unit should only be 0 or 1.")

    def set_reference_power(self) -> None:
        """Performs internal power reference. Necessary to use read_relative_power method."""
        _ret = Optotest.SetReference()
        self._check(self.get_func_name(), _ret)

    # Doesn't really work this way, since it has two distinct methods to read 
    # absolute power (read_power) and relative power (read_relative_power)    
    # def set_absolute(self) -> None:
    #     """Sets active channel to absolute (dBm) measurement mode."""
    #     _ret = Optotest.SetAbsolute()
    #     self._check(self.get_func_name(), _ret)

    def _convert_to_linear(self, value: float) -> float:
        """Converts power from [dBm] to [W]."""
        return math.pow(10, value/10 - 3) # P_lin = 10 ^ (P_dBm/10 - 3)

    def _convert_to_linear_relative(self, value: float) -> float:
        """Converts power from [dB] to adimensional (but also to [W], but it's never used in this way)."""
        return math.pow(10, value/10) # P_lin = 10 ^ (P_dB/10) 


    def _check(self, func_name: str, err_code: int) -> None:
        if Optotest.ErrorCodes(err_code) == Optotest.ErrorCodes.OK_0 or Optotest.ErrorCodes(err_code) == Optotest.ErrorCodes.OK_1:
            self.logger.debug(func_name + ":" + Optotest.ErrorCodes(err_code).name)
        else:
            self.logger.error(func_name + ":" + Optotest.ErrorCodes(err_code).name)
            raise Exception(f"{func_name}:{err_code.name}")
