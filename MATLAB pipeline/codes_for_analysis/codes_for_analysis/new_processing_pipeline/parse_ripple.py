import math
import os
import datetime
import re
import numpy as np
import psutil
import platform
import scipy.io
import shutil


# add python version of neuroshare to path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'neuroshare', 'python')))


from ns_get_analog_data_block import ns_GetAnalogDataBlock
from ns_get_analog_info import ns_GetAnalogInfo
from ns_open_file import *
import time

sys.path.append(os.path.join(os.path.dirname(__file__)))

from utils import MCW_location, get_bundle
from reylab_custompath import reylab_custompath

def parse_ripple(filenames, hours_offset=5, remove_chs=None, macro=None, 
                 max_memo_GB=None, overwrite=False, 
                 channels=None, which_system_micro='RIP'):
    # This code requires the neuroshape library in the path.
    # max_memo_GB is an idea of the number of GB allocated for the data to be
    # stored in RAM, so it is used to compute the number of segments in which
    # the data should be split for processing

    if which_system_micro is None or which_system_micro == '':
        which_system_micro = 'RIP'

    name = platform.uname()[1]
    if os.name == 'nt':
        current_user = os.environ['USERNAME']
    elif os.name == 'posix':
        current_user = os.environ['USER']

    dir_base = f'/home/{current_user}/Documents/GitHub/codes_emu'
    if os.path.exists(dir_base):
        sys.path.append(dir_base)
    custompath = reylab_custompath(['tasks/locations/'])

    if 'REYLAB' in name:
        params = MCW_location(['MCW-' + which_system_micro])
        hours_offset = params['offset']

    if hours_offset is None:
        hours_offset = 0

    if filenames is None:
        aux = os.listdir('.')
        filenames = [f for f in aux if f.endswith('.ns5')]

    expr2remove = '-\d+$'

    if macro is None:
        macro = []

    if remove_chs is None:
        remove_chs = []

    if channels is None:
        channels = []

    if os.name == 'nt':
        memo_available = int(np.floor(psutil.virtual_memory().available * 0.80))
    elif os.name == 'posix':
        memo_available = 12 * (1024) ** 3  # reduce if necessary
    else:
        memo_available = 12 * (1024) ** 3  # reduce if necessary

    if max_memo_GB is not None and max_memo_GB != '':
        max_memo = max_memo_GB * (1024) ** 3
        if max_memo > memo_available:
            raise ValueError('max_memo_GB > 80% of Physical Memory Available')
    else:
        max_memo = memo_available

    t_begin = time.time()

    if isinstance(filenames, str):
        filenames = [filenames]
    folder_path = os.path.dirname(filenames[0])
    metadata_file = os.path.join(folder_path, 'NSx.mat')
    if os.path.exists(metadata_file):
        metadata = scipy.io.loadmat(metadata_file, squeeze_me=True)
    else:
        metadata = []

    new_files = []
    for fi, filename in enumerate(filenames):
        new_files.append({'name': filename})
        if len(filename) < 3 or (not filename[1:3] == ':\\' and not filename[0] == '/' 
            and not filename[1] == '/' and not filename[0:2] == '\\\\' and not filename[1:3] == ':/'):
            filename = os.path.join(os.getcwd(), filename)

        ns_status, hFile = ns_OpenFile(filename, 'single')

        with open(filename, 'rb') as fid:
            fid.seek(294, os.SEEK_SET)
            Date = np.fromfile(fid, dtype=np.uint16, count=8)

        Start_Time = datetime.datetime(Date[0], Date[1], Date[3], Date[4], Date[5], Date[6]) - datetime.timedelta(hours=hours_offset)
        Rec_length_sec = hFile['TimeSpan'] / 30000
        End_Time = Start_Time + datetime.timedelta(seconds=Rec_length_sec)

        Date_Time = [Start_Time.strftime("%Y-%m-%d %H:%M:%S"), End_Time.strftime("%Y-%m-%d %H:%M:%S")]

        if ns_status == 'ns_FILEERROR':
            raise ValueError(f'Unable to open file: {filename}')

        ns_filetype = hFile['FileInfo'][0]['Type'][2] # 5 for ns5

        if ns_filetype == '1':
            sr = 500
        elif ns_filetype == '2':
            sr = 1000
        elif ns_filetype == '3':
            sr = 2000
        elif ns_filetype == '4':
            sr = 10000
        elif ns_filetype in ['5', '6']:
            sr = 30000
        else:
            raise ValueError(f'ERROR: {hFile.FileInfo.Type} file type not supported')

        nchan = len(hFile['Entity'])  # number of channels
        samples_per_channel = np.ceil(max_memo / (nchan * 2))
        if fi == 0:
            # get info about channels in nsx file
            chs_info = {'unit': [hFile['Entity'][i]['Units'] for i in range(nchan)],
                        'label': [hFile['Entity'][i]['Label'] for i in range(nchan)],
                        'conversion': [hFile['Entity'][i]['Scale'] for i in range(nchan)],
                        'id': [hFile['Entity'][i]['ElectrodeID'] for i in range(nchan)],
                        'pak_list': [0] * nchan,
                        'dc': [0] * nchan,
                        'macro': [hFile['Entity'][i]['Label'] for i in range(nchan)],
                        'output_name': [hFile['Entity'][i]['Label'] for i in range(nchan)]}
            micros = [i for i, x in enumerate(chs_info['unit']) if x == 'uV']
            if macro is not None and len(macro):
                chs_info['macro'] = [macro[int(np.ceil(x / 9))] if micros[i] else chs_info['macro'][i] for i, x in enumerate(range(nchan))]

            outfile_handles = [None] * nchan  # some will be empty
            _, fext = os.path.splitext(filename)
            fext = fext.lower()
            nsx_ext = fext[-1]
            ch_ext = '.NC' + nsx_ext
            if channels is None or channels == []:
                channels = hFile['FileInfo'][0]['ElectrodeList']

            remove_channels_by_label = ['(ref(.*))$']
            for ci in range(len(channels)):
                if re.search(remove_channels_by_label[0], chs_info['label'][ci]):
                    remove_chs.append(channels[ci])
            remove_chs = list(set(remove_chs))

            if remove_chs != []:
                channels = list(set(channels) - set(remove_chs))
            parsed_chs = []
            new_channel_id = []
            for i in range(nchan):
                c = hFile['FileInfo'][0]['ElectrodeList'][i]
                if c in channels:
                    ccname = c
                    if metadata != []:  # NSx file in current folder
                        repetead = [x for x in metadata['NSx'] if (x['chan_ID'] == ccname) and (x['sr'] == sr)]
                        if len(repetead):  # found channel
                            pos = [i for i, x in enumerate(repetead) if overwrite]
                            if len(pos):
                                f2delete = os.path.join(os.getcwd(), repetead[pos[0]]['output_name'] + repetead[pos[0]]['ext'])
                                print(f'Overwritting channel {metadata["NSx"][pos[0]]["chan_ID"]}. Deleting file {f2delete}')
                                if os.path.exists(f2delete):
                                    os.remove(f2delete)
                            else:
                                print(f'Skipping channel {c}, already parsed.')
                                continue  # If output_name wasn't set, the existing parsed channels won't be overwritten.
                    parsed_chs.append(c)
                    new_channel_id.append(ccname)

                    ix = chs_info['id'].index(c)
                    output_name = chs_info['macro'][ix]
                    outn_i = re.search(expr2remove, output_name)
                    if outn_i is not None and outn_i.start() > 0:
                        output_name = output_name[:outn_i.start() - 1]
                    folder_path = os.path.dirname(filename)
                    chs_info['output_name'][ix] = f'{output_name}_{ccname}'
                    nc_file_path = f"{folder_path}/{chs_info['output_name'][ix] + ch_ext}"
                    
                    outfile_handles[i] = open(nc_file_path, 'wb')

            new_files[fi]['first_sample'] = 1
        else:
            new_files[fi]['first_sample'] = new_files[fi-1]['lts'] + new_files[fi-1]['first_sample']
        if not parsed_chs:
            print('Without channels to parse.')
            return
        lts = hFile['TimeSpan']/(30000/sr)
        new_files[fi]['lts'] = lts
        N = lts
        num_segments = math.ceil(N/samples_per_channel)
        samples_per_segment = min(samples_per_channel, N)
        print(f'Data will be processed in {num_segments} segments of {samples_per_segment} samples each.')
        # min_valid_val = np.zeros((nchan, 1))
        # max_valid_val = np.zeros((nchan, 1))
        # for i in range(nchan):
        #     _, nsAnalogInfo = ns_GetAnalogInfo(hFile, i)
        #     min_valid_val[i] = nsAnalogInfo['MinVal']/nsAnalogInfo['Resolution']
        #     max_valid_val[i] = nsAnalogInfo['MaxVal']/nsAnalogInfo['Resolution']

        t_seg = time.time()
        for j in range(num_segments):
            ini = (j)*samples_per_segment
            fin = min((j+1)*samples_per_segment, N)
            _, Data = ns_GetAnalogDataBlock(hFile, list(range(0, nchan)), ini, fin-ini, 'unscale')
            for i in range(nchan):
                if outfile_handles[i]:
                    if sr == 30000:
                        pak_lost = Data[:,i] == np.iinfo(np.int16).min
                        Data[pak_lost,i] = 0
                        chs_info['pak_list'][i] = chs_info['pak_list'][i] + np.sum(pak_lost)
                    else:
                        pak_lost = Data[:,i] == np.finfo(np.float32).max
                        Data[pak_lost,i] = 0
                        chs_info['pak_list'][i] = chs_info['pak_list'][i] + np.sum(pak_lost)
                    Data_ch = Data[:,i].astype(float)
                    if sr != 30000:
                        chs_info['dc'][i] = (np.max(Data_ch) + np.min(Data_ch))/2
                        Data_ch = Data_ch - chs_info['dc'][i]
                        chs_info['conversion'][i] = np.max(np.abs(Data_ch))/np.iinfo(np.int16).max
                        Data_ch = np.round(Data_ch/chs_info['conversion'][i])
                    outfile_handles[i].write(Data_ch.astype(np.int16).tobytes())
            t_seg_mins, t_seg_secs = divmod(time.time() - t_seg, 60)
            print(f'Segment {j+1} out of {num_segments} processed in {t_seg_mins:.1f} mins and {t_seg_secs:.2f} secs.')
            t_seg = time.time()
        
        for fh in outfile_handles:
            if fh is not None:
                fh.close()
        if not metadata:
            dtype = np.dtype([('chan_ID', 'O'), ('conversion', 'O'), ('dc', 'O'), 
                              ('label', 'O'), ('bundle', 'O'), ('is_micro', 'O'), 
                              ('unit', 'O'), ('electrode_ID', 'O'), ('which_system', 'O'), 
                              ('ext', 'O'), ('lts', 'O'), ('filename', 'O'), 
                              ('sr', 'O'), ('output_name', 'O'), ('pak_list', 'O')])
            NSx = np.zeros((len(new_channel_id),), dtype=dtype)
            files_dtype = np.dtype([('name', 'O'), ('first_sample', 'O'), ('lts', 'O')])
            files = np.zeros((len(new_files),), dtype=files_dtype)
        else:
            NSx = metadata['NSx']
            files = metadata['files'].flatten()
            old_NSx_shape = NSx.shape[0]

        lts = sum([new_files[i]['lts'] for i in range(len(new_files))])
        print(f'{lts} data points written per channel')

        for ci in range(len(new_channel_id)):
            ch = new_channel_id[ci]
            elec_id = parsed_chs[ci]
            
            ix = np.where(chs_info['id']==elec_id)[0][0]
            info = {
                'chan_ID': ch,
                'conversion': chs_info['conversion'][ix],
                'dc': chs_info['dc'][ix],
                'label': chs_info['label'][ix],
                'bundle': get_bundle(chs_info['macro'][ix]),
                'is_micro': chs_info['label'][ix][0] == 'm',
                'unit': chs_info['unit'][ix],
                'electrode_ID': elec_id,
                'which_system': 'RIPPLE',
                'ext': ch_ext,
                'lts': lts,
                'filename': filenames,
                'sr': sr,
                'output_name': chs_info['output_name'][ix],
                'pak_list': chs_info['pak_list'][ix]
            }
            nsx_idx = None
            for i, nsx in enumerate(NSx):
                if nsx['chan_ID']==ch and nsx['sr']==sr:
                    # replace nsx with info
                    nsx_idx = i
                    copy_info(info, nsx)
                    break
            
            if nsx_idx is not None:
                print(f'Updated channel {ch} info.')
            else:
                if metadata: #metadata existed
                    ind = ci + old_NSx_shape
                    new_size = ind + 1  # New size must be large enough
                    dtype = np.dtype([('chan_ID', 'O'), ('conversion', 'O'), ('dc', 'O'), 
                              ('label', 'O'), ('bundle', 'O'), ('is_micro', 'O'), 
                              ('unit', 'O'), ('electrode_ID', 'O'), ('which_system', 'O'), 
                              ('ext', 'O'), ('lts', 'O'), ('filename', 'O'), 
                              ('sr', 'O'), ('output_name', 'O'), ('pak_list', 'O')])
                    new_NSx = np.empty(new_size, dtype=dtype)  # Create new array
                    new_NSx[: NSx.shape[0]] = NSx  # Copy existing data
                    NSx = new_NSx
                else:
                    ind = ci
                
                copy_info(info, NSx[ind])

        for i in range(len(new_files)):
            file_info = {
                'name': new_files[i]['name'],
                'first_sample': new_files[i]['first_sample'],
                'lts': new_files[i]['lts']
            }
            file_idx = None
            for idx, file in enumerate(files):
                if file['name'] == new_files[i]['name']:
                    file_idx = idx
                    copy_files(new_files[i], file)
                    break
            
            if file_idx is not None:
                print(f'Updated file {new_files[i]["name"]} info.')
            else:
                copy_files(new_files[i], files[i])

        freq_priority = [30000, 7500,2000, 10000, 1000, 500]

        metadata = {'NSx': NSx, 'files': files, 'freq_priority': freq_priority, 'Date_Time': Date_Time}

        scipy.io.savemat(metadata_file, metadata)

        time_taken = time.time() - t_begin
        t_mins, t_secs = divmod(time_taken, 60)
        print(f'{os.path.basename(filenames[fi])} parsed in {t_mins:.1f} minutes and {t_secs:.2f} seconds.')
        
        custompath.rm()

def copy_info(frm, to):
    to['chan_ID'] = frm['chan_ID']
    to['conversion'] = frm['conversion']
    to['dc'] = frm['dc']
    to['label'] = frm['label']
    to['bundle'] = frm['bundle']
    to['is_micro'] = frm['is_micro']
    to['unit'] = frm['unit']
    to['electrode_ID'] = frm['electrode_ID']
    to['which_system'] = frm['which_system']
    to['ext'] = frm['ext']
    to['lts'] = frm['lts']
    to['filename'] = frm['filename']
    to['sr'] = frm['sr']
    to['output_name'] = frm['output_name']
    to['pak_list'] = frm['pak_list']

def copy_files(frm, to):
    to['name'] = frm['name']
    to['first_sample'] = frm['first_sample']
    to['lts'] = frm['lts']

if __name__ == '__main__':
    
    # file_path = os.path.join(os.getcwd(), 'ns_Data', 'EMU-001_subj-MCW-FH_test_task-gaps\EMU-001_subj-MCW-FH_test_task-gaps_run-01_RIP.ns5')
    #file_path = os.path.join(os.getcwd(), 'ns_Data', 'EMU-004_subj-MCW-FH_002_task-gaps_run-01_RIP.ns5')
    #if os.path.exists(file_path):
    #    parse_ripple(file_path, overwrite=True)
    #else:
    #    print(f'File {file_path} not found.')
    file = '/mnt/data0/sEEG_DATA/MCW-FH_006/EMU/EMU-015_subj-MCW-FH_006_task-gaps/EMU-001_subj-MCW-FH_006_task-gaps_run-04_RIP.nf3'
    folder_name = os.path.dirname(file) # Folder where the orig .nf3 resides
    run_folder_name = os.path.basename(file)[:-8] # New folder name (run-xx)
    run_folder_path = os.path.join(folder_name, run_folder_name) # New folder path
    os.makedirs(run_folder_path, exist_ok=True) # Create new folder with new folder path
    
    nf3_file = os.path.join(run_folder_path, os.path.basename(file)) # Path of the nf3 copy
    if not os.path.exists(nf3_file):
        shutil.copy2(file, run_folder_path)

    parse_ripple(filenames=nf3_file, overwrite=True)