from scipy.io import loadmat
from os import path
import numpy as np 

def  read_nsx_file(fullfile):
    folder = path.dirname(fullfile)
    metadata = loadmat(path.join(folder, 'NSx.mat'),variable_names='NSx', squeeze_me=True, simplify_cells=True)['NSx']
    [filename, ext] = path.splitext(path.basename(fullfile))
    
    info = list(filter(lambda x: (x['output_name'] == filename) and (x['ext'] == ext), metadata))
    if len(info) == 0:
        assert info is not None, 'file: {} not found in nsx'.format(filename)
    info = info[0]
    gain = info['conversion']

    if 'dc' in info:
        dc = info['dc']
    else:
        dc=0
    x_raw = np.fromfile(fullfile, 'int16') * gain+dc

 
    return x_raw, info
