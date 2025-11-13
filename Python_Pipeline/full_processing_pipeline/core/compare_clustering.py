import numpy as np
from scipy.io import loadmat
import os
import glob
import re
import pywaveclus.spike_detection as sd
import pywaveclus.feature_extraction as fe
import pywaveclus.clustering as clu
from spclustering import plot_temperature_plot
import matplotlib.pyplot as plt
import re

class compare():
    def __init__(self) -> None:
        pass
    def compare_clustering(self,directory,channel_list):
        features = {}
        tmp_files = glob.glob(directory+'/tmp*')
        tmp_files = sorted(tmp_files)
        for i,tmp_file in enumerate(tmp_files):
            with open(tmp_file, 'r') as file:
                lines = file.readlines()

            # Process the data
            data = []
            for line in lines:
                # Convert each line to a list of floats
                row = list(map(float, line.split()))
                data.append(row)

            # Convert the list of lists to a NumPy array
            np_array = np.array(data)
            features[channel_list[i]] = np_array
        labels, metadata = clu.SPC_clustering(features)
        print('lets see if this works')

if __name__ == '__main__':
    directory = '/mnt/data0/Python_pipeline_results/matlab_results/patient 7 EMU002'
    filename_list = glob.glob(directory+'/*spikes.mat')
    channel_list = []
    for filename in filename_list:
        pattern = r'\d+\.?\d*'
        matches = re.findall(pattern, filename)
        channel_list.append(matches[-1])
    channel_list = np.concatenate((np.arange(257,264+1),np.arange(298,305+1),np.arange(321,328+1),np.arange(266,273+1),np.arange(330,337+1),np.arange(289,296+1)),axis=0)
    com = compare()
    com.compare_clustering(directory=directory,channel_list=channel_list)
    