from hmac import new
import math
import os
import datetime
import pickle
import re
import threading
import numpy as np
import psutil
import platform
import scipy.io
import time
import concurrent.futures

# add python version of neuroshare to path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'neuroshare/pyns')))
from nsfile import NSFile
from nsentity import EntityType 

core_path = os.path.join(os.path.dirname(__file__))
sys.path.append(core_path)

from utils import MCW_location, get_bundle
from reylab_custompath import reylab_custompath

def process_entity_data(entity, ns_file_data, num_segments, overwrite=True):
    ch_ext = '.NC' + ns_file_data.extension[-1]
    nc_filename = f'{entity.label}_{entity.electrode_id}{ch_ext}'
    nc_filepath = os.path.join(os.path.dirname(ns_file_data.parser.fid.name), nc_filename)
    if os.path.exists(nc_filepath) and overwrite:
        os.remove(nc_filepath)
    elif os.path.exists(nc_filepath) and not overwrite:
        print(f'File {nc_filepath} already exists. Skipping.')
        return
    
    file_pointer = open(nc_filepath, 'wb')
    seg_idx = 0
    new_seg_samples_count = math.ceil(ns_file_data.parser.n_data_points / num_segments)
    for data_idx in range(0, ns_file_data.parser.n_data_points, new_seg_samples_count):
        ch_seg_data = ns_file_data.parser.get_analog_data(channel_index=entity.electrode_id, start_index=data_idx, index_count=new_seg_samples_count)
        if ns_file_data.parser.timestamp_resolution == 30000:
            pak_lost = ch_seg_data == np.iinfo(np.int16).min
            ch_seg_data[pak_lost] = 0
            # chs_info['pak_list'][entity.electrode_id] = chs_info['pak_list'][entity.electrode_id] + np.sum(pak_lost)
        else:
            pak_lost = ch_seg_data == np.finfo(np.float32).max
            ch_seg_data[pak_lost] = 0
            # chs_info['pak_list'][entity.electrode_id] = chs_info['pak_list'][entity.electrode_id] + np.sum(pak_lost)
        file_pointer.write(ch_seg_data.tobytes())
        print(f'Channel {entity.label} {seg_idx+1}/{num_segments} processed.')
        seg_idx += 1

    file_pointer.close()

def parse_ripple(filenames, hours_offset=5, max_ram_to_use_GB=0.1, 
                 overwrite=True, system='RIP'):
    
    begin_time = time.time()

    name = platform.uname()[1]
    if os.name == 'nt':
        current_user = os.environ['USERNAME']
    elif os.name == 'posix':
        current_user = os.environ['USER']
    else:
        raise ValueError('OS not supported')
    dir_base = f'/home/{current_user}/Documents/GitHub/codes_emu'
    if os.path.exists(dir_base):
        sys.path.append(dir_base)
    custompath = reylab_custompath(['tasks/locations/'])

    if 'REYLAB' in name:
        params = MCW_location(['MCW-' + system])
        hours_offset = params.offset

    if filenames is None:
        aux = os.listdir('.')
        filenames = [f for f in aux if f.endswith('.ns5')]

    usable_memory_bytes = int(np.floor(psutil.virtual_memory().available * 0.80))

    if max_ram_to_use_GB is not None:
        usable_memory_bytes = max_ram_to_use_GB * 1024**3
    else:
        max_ram_to_use_GB = usable_memory_bytes / 1024**3

    if isinstance(filenames, str):
        filenames = [filenames]

    metadata_file = os.path.join(os.path.dirname(filenames[0]), 'NSx.mat')
    if os.path.exists(metadata_file):
        metadata = scipy.io.loadmat(metadata_file, squeeze_me=True)
    else:
        metadata = []

    new_files = []
    for file_idx, filename in enumerate(filenames):
        new_files.append({'name': filename})

        ns_file = NSFile(filename)
        ns_file_info = ns_file.get_file_info()
        ns_file_data = ns_file.get_file_data(ext='ns5')

        num_segments = math.ceil((ns_file_data.parser.size) / usable_memory_bytes)  
        print(f'Dividing data ({ns_file_data.parser.size / 1024**3:.2f} GB) into \
              {num_segments} segments of {max_ram_to_use_GB:.2f} GB each. \
              Each channel file size: {max_ram_to_use_GB/ns_file_data.parser.channel_count:.3f} GB.')


        if file_idx == 0:
            # get info about channels in nsx file
            chs_info = {'unit': [entity.units for entity in ns_file.entities],
                        'label': [entity.label for entity in ns_file.entities],
                        'conversion': [entity.scale for entity in ns_file.entities],
                        'id': [entity.electrode_id for entity in ns_file.entities],
                        'pak_list': [0] * ns_file_info.entity_count,
                        'dc': [0] * ns_file_info.entity_count,
                        'macro': [entity.label for entity in ns_file.entities if entity.units != 'uV'],
                        'output_name': [entity.label for entity in ns_file.entities]}

        entity_threads = []
        for entity in ns_file.entities:
            if -1 != entity.label.find('raw'):
                process_entity_data(entity, ns_file_data, num_segments, overwrite=overwrite)

        if not metadata:
            files = []
            NSx = []
        else:
            NSx = metadata['NSx']
            files = metadata['files']


        freq_priority = [30000, 2000, 10000, 1000, 500]
        metadata = {'NSx': NSx, 'files': files, 'freq_priority': freq_priority, 'Date_Time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        
        if os.name == 'posix':
            custompath.rm()

        total_time = time.time() - begin_time
        hours, mins, secs = total_time // 3600, (total_time % 3600) // 60, total_time % 60
        print(f'Total time spent in parsing the data was {hours:.0f} hours, {mins:.0f} minutes and {secs:.0f} seconds.')



if __name__ == '__main__':
    file_path = os.path.join(os.getcwd(), 'ns_Data', 'EMU-001_subj-MCW-FH_test_task-gaps/EMU-001_subj-MCW-FH_test_task-gaps_run-01_RIP.ns5')
    parse_ripple(file_path)