import math
import os
import struct
import numpy
from scipy.io import loadmat


class NCxFiles:
    # class variables

    def __init__(self, NSx_file):
        mat_contents = loadmat(NSx_file)
        self.NSx = mat_contents['NSx'][0].squeeze()
        chan_1 = self.NSx[0]
        lts = chan_1[10][0][0]
        self.sample_rate = chan_1[12][0][0]
        self.rec_length = lts/self.sample_rate  # in seconds

    # filter function or add to read channels/bundles
    def read_channels(self, tmin, selected_channels=None,selected_chan_IDs=None):
        # number take as channel ID
        # string take as label
        # default tmin is 10
        raw_dict = {}
        power_dict = {}
        rows = self.NSx
        selected_rows = []
        selected_NCx_files = []
        for row in rows:
            channel = row[3]  # channel label
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
            if selected_chan_IDs == None and selected_channels != None:
                for selected_channel in selected_channels:
                    if str(selected_channel) in str(channel):
                        selected_rows.append(row)
                        selected_NCx_files.append(NCx_file)
                    else:
                        continue
            elif selected_chan_IDs != None and selected_channels == None:
                selected_channels = []
                for selected_chan_ID in selected_chan_IDs:
                    if str(selected_chan_ID) in str(chan_ID):
                        selected_rows.append(row)
                        selected_NCx_files.append(NCx_file)
                        selected_channels.append(channel)
                    else:
                        continue
                
        zippy = zip(selected_NCx_files, selected_rows, selected_channels)
        # print(bundles)
        for NCx_file, selected_row, selected_channel in zippy:
            min_record = self.sample_rate * tmin
            max_record = math.floor(
                min_record + self.sample_rate * self.rec_length)
            tmax = max_record / self.sample_rate
            # Open the binary file for reading in binary mode ('rb')
            with open(NCx_file, 'rb') as file:
                # Read the binary data from the file. Start at begining of data with
                # seeek()
                binary_data = file.read()
                file.seek((min_record - 1) * 2)
            # Create a format string for little-endian unsigned 16-bit integers

            format_string = '<' + 'h' * (len(binary_data) // 2)
            # Unpack the binary data into a list of integers
            unpacked_data = struct.unpack(format_string, binary_data)
            # update raw dictionary and power dictionary. Raw dictionary will be used
            # for 2D graphs, and Power dictionary will be normalized and used for the
            # 3D animation
            key = selected_channel
            conversion = selected_row[1][0][0]
            dc_offset = selected_row[2][0][0]
            raw_samples = [x * conversion + dc_offset
                           for x in unpacked_data[min_record - 1:max_record]]
            power_samples = [raw_samples[i] **
                             2 for i in range(len(raw_samples))]
            power_dict.update({key: power_samples})
            raw_dict.update({key: raw_samples})
        return power_dict, raw_dict

    def read_bundles(self, tmin, selected_bundles):
        raw_dict = {}
        power_dict = {}
        rows = self.NSx
        selected_NCx_files = []
        selected_rows = []
        selected_channels = []
        for row in rows:
            filepath = row[11][0]
            directory = os.path.dirname(filepath)
            directory = str(directory)
            label = row[3][0]
            label = str(label)
            ext = row[9][0]
            ext = str(ext)
            chan_ID = row[0][0][0]
            chan_ID = str(chan_ID)
            NCx_file = directory + '/' + label + '_' + chan_ID + ext
            bundle = row[4]
            selected_channel = row[3]
            for selected_bundle in selected_bundles:
                if str(selected_bundle) in str(bundle):
                    selected_rows.append(row)
                    selected_channels.append(selected_channel)
                    selected_NCx_files.append(NCx_file)
                else:
                    continue
        zippy = zip(selected_NCx_files, selected_rows, selected_channels)
        for NCx_file, selected_row, selected_channel in zippy:
            min_record = self.sample_rate * (tmin+10)
            max_record = math.floor(
                min_record + self.sample_rate * self.rec_length)
            tmax = max_record / self.sample_rate
            # Open the binary file for reading in binary mode ('rb')
            with open(NCx_file, 'rb') as file:
                # Read the binary data from the file. Start at begining of data with
                # seeek()
                binary_data = file.read()
                file.seek((min_record - 1) * 2)
            # Create a format string for little-endian unsigned 16-bit integers
            if self.sample_rate == 2000:
                format_string = '<' + 'h' * (len(binary_data) // 2)
            else:
                format_string = '<' + 'f' * (len(binary_data) // 4)
            # Unpack the binary data into a list of integers
            unpacked_data = struct.unpack(format_string, binary_data)
            # update raw dictionary and power dictionary. Raw dictionary will be used
            # for 2D graphs, and Power dictionary will be normalized and used for the
            # 3D animation
            key = selected_channel[0][:4]
            conversion = selected_row[1][0][0]
            dc_offset = selected_row[2][0][0]
            raw_samples = [x * conversion + dc_offset
                           for x in unpacked_data[min_record - 1:max_record]]
            power_samples = [raw_samples[i] **
                             2 for i in range(len(raw_samples))]
            power_dict.update({key: power_samples})
            raw_dict.update({key: raw_samples})
        return power_dict, raw_dict
