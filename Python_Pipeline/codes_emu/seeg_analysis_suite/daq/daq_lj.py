"""
# Author: Sunil Mathew
# Date: 2/28/2024
# Send pulses whenever a stimuli is shown from U3 DAQ from 
# LabJack (https://labjack.com/blogs/faq/getting-started-1) 
# connected via USB. Pulses are receieved by Ripple explorer 
# which is connected to the Ripple Summit NIP
"""
import sys
from datetime import datetime

import u3

import time
import traceback

class DAQ_LJ:

    def __init__(self):

        try:

            self.d = u3.U3()
            # Set EIO0-EIO7 to output
            print(self.d.configU3(EIODirection=0xFF))
            self.b_init = True

        except:
            print(traceback.format_exc())

    def send_info(self, info):
        """
        Send info from DAQ to Ripple.
        U3 has a DB15 port. We are using EIO0-EIO7 for digital output which are 8-15"""
        if self.b_init:
            try:
                self.d.debug = True
                self.d.getFeedback(u3.BitStateWrite(IONumber=8, State=info))
                # self.d.getFeedback(u3.BitStateWrite(IONumber=8, State=(info & 0x01)))
            except:
                print(traceback.format_exc())

    def close(self):
        self.d.close()


if __name__ == '__main__':
    lines_onoff = 13
    blank_on = 11
    lines_flip_blank = 103
    lines_flip_pic = 22
    trial_on = 26
    data_signature_on = 64
    data_signature_off = 128

    daq_lj = DAQ_LJ()
    daq_lj.send_info(data_signature_on)
    time.sleep(0.05)
    daq_lj.send_info(data_signature_off)
    time.sleep(0.45)
    daq_lj.send_info(data_signature_on)
    time.sleep(0.05)
    daq_lj.send_info(data_signature_off)
    time.sleep(0.45)
    daq_lj.send_info(data_signature_on)
    time.sleep(0.05)
    daq_lj.send_info(data_signature_off)

    # Send 10 pulses 500ms apart 
    for i in range(10):
        daq_lj.send_info(0x00)
        daq_lj.send_info(0xFF)
        time.sleep(0.5)

    daq_lj.close()