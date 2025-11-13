"""
# Author: Sunil Mathew
# Date: 2/28/2024
# Send pulses whenever a stimuli is shown from 1280FS DAQ from MCC mccdaq.com 
# connected via USB. Pulses are receieved by Ripple explorer 
# which is connected to the Ripple Summit NIP
"""

from __future__ import absolute_import, division, print_function
import time
import traceback
from sys import platform

try:
    if platform == "win32":
        from mcculw import ul
        from mcculw.enums import DigitalIODirection, InterfaceType
        from mcculw.device_info import DaqDeviceInfo
except:
    print(traceback.format_exc())



class DAQ_MC:

    def __init__(self):
        # By default, the example detects and displays all available devices and
        # selects the first device listed. Use the dev_id_list variable to filter
        # detected devices by device ID (see UL documentation for device IDs).
        # If use_device_detection is set to False, the board_num variable needs to
        # match the desired board number configured with Instacal.
        self.use_device_detection = True
        dev_id_list = []
        self.board_num = 0
        self.b_init = False
        self.output_ports = []

        try:
            if platform == "linux" or platform == "linux2" or platform == "darwin":
                print(f'DAQ MC not supported on {platform}')
                return
            if self.use_device_detection:
                self.config_first_detected_device(self.board_num, dev_id_list)

            self.daq_dev_info = DaqDeviceInfo(self.board_num)
            if not self.daq_dev_info.supports_digital_io:
                raise Exception('Error: The DAQ device does not support '
                                'digital I/O')

            print('\nActive DAQ device: ', self.daq_dev_info.product_name, ' (',
                self.daq_dev_info.unique_id, ')\n', sep='')

            self.dio_info = self.daq_dev_info.get_dio_info()

            # DAQ 1208FS has 4 ports, each port has 8 bits
            # We'll use port A and B for digital output
            for port in self.dio_info.port_info:
                if port.supports_output:
                    # If the port is configurable, configure it for output.
                    if port.is_port_configurable:
                        ul.d_config_port(self.board_num, port.type, DigitalIODirection.OUT)
                        self.output_ports.append(port)
                        print('Port: ', port.type.name, ' configured for output')
            if len(self.output_ports) == 0:
                print('Error: The DAQ device does not support '
                                'digital output')
            else:
                self.b_init = True
        except:
            print(traceback.format_exc())

    def config_first_detected_device(self, board_num, dev_id_list=None):
        """Adds the first available device to the UL.  If a types_list is specified,
        the first available device in the types list will be add to the UL.

        Parameters
        ----------
        board_num : int
            The board number to assign to the board when configuring the device.

        dev_id_list : list[int], optional
            A list of product IDs used to filter the results. Default is None.
            See UL documentation for device IDs.
        """
        ul.ignore_instacal()
        devices = ul.get_daq_device_inventory(InterfaceType.ANY)
        if not devices:
            raise Exception('Error: No DAQ devices found')

        print('Found', len(devices), 'DAQ device(s):')
        for device in devices:
            print('  ', device.product_name, ' (', device.unique_id, ') - ',
                'Device ID = ', device.product_id, sep='')

        device = devices[0]
        if dev_id_list:
            device = next((device for device in devices
                        if device.product_id in dev_id_list), None)
            if not device:
                err_str = 'Error: No DAQ device found in device ID list: '
                err_str += ','.join(str(dev_id) for dev_id in dev_id_list)
                raise Exception(err_str)

        # Add the first DAQ device to the UL with the specified board number
        ul.create_daq_device(board_num, device)

    def send_info(self, info):

        if self.b_init:
            try:
                for port in self.output_ports:
                    ul.d_out(self.board_num, port.type, info)
            except:
                print(traceback.format_exc())

    def close(self):
        if self.use_device_detection:
            ul.release_daq_device(self.board_num)


if __name__ == '__main__':
    lines_onoff = 13
    blank_on = 11
    lines_flip_blank = 103
    lines_flip_pic = 22
    trial_on = 26
    data_signature_on = 64
    data_signature_off = 128

    daq_mc = DAQ_MC()
    daq_mc.send_info(data_signature_on)
    time.sleep(0.05)
    daq_mc.send_info(data_signature_off)
    time.sleep(0.45)
    daq_mc.send_info(data_signature_on)
    time.sleep(0.05)
    daq_mc.send_info(data_signature_off)
    time.sleep(0.45)
    daq_mc.send_info(data_signature_on)
    time.sleep(0.05)
    daq_mc.send_info(data_signature_off)

    # Send 10 pulses 500ms apart 
    for i in range(10):
        daq_mc.send_info(0x00)
        daq_mc.send_info(0xFF)
        time.sleep(0.5)

    daq_mc.close()