import math
import os
import struct
import numpy
from scipy.io import loadmat
import numpy as np


class NCxFiles:
    # class variables

    def __init__(self, NSx_file):
        self.NSx_file = NSx_file
        mat_contents = loadmat(self.NSx_file)
        self.NSx = mat_contents['NSx'][0].squeeze()
        chan_1 = self.NSx[0]
        lts = chan_1[10][0][0]
        self.sample_rate = chan_1[12][0][0]
        self.rec_length = lts/self.sample_rate  # in seconds

    # filter function or add to read channels/bundles
    def read_channels(self, tmin, tmax,selected_channels):
        # number take as channel ID
        # string take as label
        # default tmin is 10
        raw_dict = {}
        power_dict = {}
        rows = self.NSx
        selected_rows = []
        selected_NCx_files = []
        units = []
        for i,row in enumerate(rows):
            if len(selected_channels) == len(selected_rows):
                break
            if selected_channels[i].isdigit():
                channel = row[0][0]
            else:
                channel = row[3][0]  # channel label
            filepath = str(row[11][0])
            directory = os.path.dirname(self.NSx_file)
            directory = str(directory)
            label = row[3][0]
            label = str(label)
            ext = row[9][0]
            ext = str(ext)
            chan_ID = row[0][0][0]
            chan_ID = str(chan_ID)
            NCx_file = str(directory + '/' + label + '_' + chan_ID + ext)

            for j,selected_channel in enumerate(selected_channels):
                try:
                    if str(selected_channel) in str(channel):
                        selected_rows.append(row)
                        selected_NCx_files.append(NCx_file)
                    else:
                        continue
                except:
                    if int(selected_channel) in int(channel):
                        selected_rows.append(row)
                        selected_NCx_files.append(NCx_file)
                    else:
                        continue
        zippy = zip(selected_NCx_files, selected_rows, selected_channels)
        # print(bundles)
        for NCx_file, selected_row, selected_channel in zippy:
            unit = selected_row[6][0]
            min_record = self.sample_rate * tmin
            max_record = math.floor(self.sample_rate * tmax)
            
            # Open the binary file for reading in binary mode ('rb')
            key = selected_channel
            conversion = selected_row[1][0][0]
            dc_offset = selected_row[2][0][0]

            f1 = open(NCx_file,'rb')
            binary_data = f1.read()
            format_string = '<' + 'h' * (len(binary_data) // 2)
            unpacked_data = struct.unpack(format_string, binary_data)
            x = np.array(unpacked_data)*conversion+dc_offset

            raw_samples = x[min_record:max_record]
            power_samples = np.abs(x[min_record:max_record])
            power_dict.update({key: power_samples})
            raw_dict.update({key: raw_samples})
            units.append(unit)
        return power_dict, raw_dict, unit

    def read_bundles(self, tmin,tmax, selected_bundles):
        raw_dict = {}
        power_dict = {}
        rows = self.NSx
        selected_NCx_files = []
        selected_rows = []
        selected_channels = []
        units = []
        for row in rows:
            filepath = str(row[11][0])
            directory = str(os.path.dirname(self.NSx_file))
            label = row[3][0]
            label = str(label)
            ext = row[9][0]
            ext = str(ext)
            chan_ID = row[0][0][0]
            chan_ID = str(chan_ID)
            NCx_file = directory + '/' + label + '_' + chan_ID + ext
            bundle = row[4][0]
            selected_channel = row[3]
            
            for selected_bundle in selected_bundles:
                if str(bundle) in str(selected_bundle):
                    selected_rows.append(row)
                    selected_channels.append(selected_channel)
                    selected_NCx_files.append(NCx_file)
                else:
                    continue
        zippy = zip(selected_NCx_files, selected_rows, selected_channels)
        for NCx_file, selected_row, selected_channel in zippy:
            unit = selected_row[6][0]
            min_record = self.sample_rate * tmin
            max_record = math.floor(self.sample_rate * tmax)
            # Open the binary file for reading in binary mode ('rb')
            key = selected_channel[0]
            conversion = selected_row[1][0][0]
            dc_offset = selected_row[2][0][0]

            f1 = open(NCx_file,'rb')
            binary_data = f1.read()
            format_string = '<' + 'h' * (len(binary_data) // 2)
            unpacked_data = struct.unpack(format_string, binary_data)
            x = np.array(unpacked_data)*conversion+dc_offset

            raw_samples = x[min_record:max_record]
            power_samples = np.abs(x[min_record:max_record])
            power_dict.update({key: power_samples})
            raw_dict.update({key: raw_samples})
            units.append(unit)
        return power_dict, raw_dict, units
