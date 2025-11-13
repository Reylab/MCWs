import os
import re
import time
import pickle
import sys
#import winreg
if sys.platform.startswith("win"):
    import winreg
from ofunctions.file_utils import get_files_recursive
from collections import OrderedDict

if os.name == 'nt':
    import winreg

def orderfields(s1, s2=None):
    """
    Order fields of a dictionary-like object.

    s1 : dict-like
        The dictionary-like object to be reordered.
    s2 : list or tuple, optional
        The order of fields to be used. If not provided, the fields will be
        sorted in ASCII dictionary order.

    Returns
    -------
    OrderedDict
        A new ordered dictionary with the fields in the specified order.

    Raises
    ------
    ValueError
        If s2 is provided and contains fields not present in s1.

    """
    if s2 is None:
        # Sort fields in ASCII dictionary order
        sorted_fields = sorted(s1.keys())
    else:
        # Use specified field order
        if set(s2) != set(s1.keys()):
            raise ValueError("s2 contains fields not present in s1")
        sorted_fields = s2

    # Create new ordered dictionary with fields in specified order
    return OrderedDict([(field, s1[field]) for field in sorted_fields])

def save_log(params, folder_name, logfile, n_attempts, MEnew):
    if params['with_acq_folder']:
        files = f'EMU-{params["EMU_num"]:.3f}_subj-{params["sub_ID"]}_task-gaps_run-*'
        os.mkdir(folder_name)
        time.sleep(5)
        os.rename(os.path.join(params['acq_remote_folder_in_beh'], files), os.path.join(folder_name, files))
        with open(os.path.join(folder_name, f'EMU-{params["EMU_num"]:.3f}_subj-{params["sub_ID"]}_task-gaps_logfile.mat'), 'wb') as f:
            pickle.dump({'logfile': logfile, 'n_attempts': n_attempts, 'MEnew': MEnew}, f)

def test_remote_folder(remote_folder):
    if os.name == 'nt':
        status = os.path.isdir(remote_folder)
    elif os.name == 'posix':
        status = False
        with open('/proc/mounts', 'r') as f:
            for line in f:
                if remote_folder in line:
                    status = True
                    break
    else:
        raise ValueError('function test_acq_folder not tested for mac')
    return status

def get_bundle(label):
    tt = re.search('\d+', label)
    if tt:
        bundle = label[:tt.start()]
    else:
        bundle = label
    return bundle

def location_setup(location):
    parts = location.split('-')
    prev_dir = os.getcwd()
    if not parts:
        raise ValueError('location must have the form: PLACE[-OPTIONAL STRINGS]')

    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'locations'))

    if parts[0] == 'MCW':
        params = MCW_location(location)
    elif parts[0] == 'BCM':
        params = BCM_location(location)
    else:
        os.chdir(prev_dir)
        raise ValueError('Location not found')

    os.chdir(prev_dir)
    return params

def MCW_location(location):
    params = {}
    params['location'] = location

    if '-WPTB' in location:
        params['windowRect'] = [0, 0, 1920, 1080]

    if os.name == 'nt':
        params['ptb_priority_normal'] = 0
        params['ptb_priority_high'] = 1
        paths = ['C:\\Program Files\\Blackrock Microsystems\\NeuroPort Windows Suite',
                 'C:\\Program Files (x86)\\Blackrock Microsystems\\Cerebus Windows Suite',
                 'C:\\Program Files (x86)\\Blackrock Microsystems\\NeuroPort Windows Suite']
        for path in paths:
            if os.path.isdir(path):
                params['cbmex_path'] = os.pathsep.join([params.get('cbmex_path', ''), path])
                break
        
        xippmex_path = r'C:\Program Files (x86)\Ripple\Trellis\Tools\xippmex'

        if os.path.exists(xippmex_path):
            params['xippmex_path'] = xippmex_path
        else:
            print("Looks like Trellis/xippmex is not installed. Please install Trellis and try again.")

    elif os.name == 'posix':
        params['ptb_priority_normal'] = 0
        params['ptb_priority_high'] = 1
        params['xippmex_path'] = '/opt/Trellis/Tools/xippmex'

    params['additional_pics'] = os.path.join('pics_space', 'pics_now')
    params['keyboards'] = ['Microsoft Microsoft® 2.4GHz Transceiver v8.0']
    params['lang'] = 'english'
    params['beh_rec_metadata'] = '/home/user/share/experimental_files/rec_metadata'
    params['acq_remote_folder_in_beh'] = '/media/acq'
    params['acq_remote_folder_in_processing'] = '/media/acq'
    params['with_acq_folder'] = True
    params['acq_is_beh'] = False
    params['acq_is_processing'] = False
    params['copy_backup'] = True
    params['ttl_device'] = 'LJ'
    params['use_photodiodo'] = True
    params['device_resp'] = 'gamepad'

    if location == 'MCW-BEH-RIP':
        params['beh_machine'] = 1
        params['proccesing_machine'] = 2
        params['trellis_data_path'] = 'C:\\Users\\user\\Trellis\\datafiles'
        params['additional_paths'] = [params.get('additional_paths', []), params['xippmex_path']]
        params['use_daq'] = True
        params['root_processing'] = '/home/user/share/experimental_files/'
        params['root_beh'] = '/home/user/share/experimental_files/'
        params['pics_root_beh'] = '/home/user/share/experimental_files/pics'
        params['pics_root_processing'] = '/home/user/share/experimental_files/pics'
        params['mapfile'] = '*.map'
        params['processing_rec_metadata'] = '/home/user/share/experimental_files/rec_metadata'
        params['early_figures_full_path'] = '/home/user/share/early_figures'
        params['system'] = 'RIP'
        params['online_notches'] = True
        params['copy_backup'] = False
        return params

    params['trellis_data_path'] = 'C:\\Users\\user\\Trellis\\datafiles'
    params['acq_network'] = True
    params['additional_paths'] = [params.get('additional_paths', []), params['xippmex_path']]
    params['backup_path'] = '/mnt/acq-hdd'
    params['use_daq'] = True
    params['root_processing'] = '/home/user/share/experimental_files/'
    params['root_beh'] = '/home/user/share/experimental_files/'
    params['pics_root_beh'] = '/home/user/share/experimental_files/pics'
    params['pics_root_processing'] = '/home/user/share/experimental_files/pics'
    params['mapfile'] = '*.map'
    params['processing_rec_metadata'] = '/home/user/share/experimental_files/rec_metadata'
    params['early_figures_full_path'] = '/home/user/share/early_figures'

    location = location[0]
    if -1 != location.find('-SS'):
        params['beh_machine'] = 'BEH-REYLAB'
        params['proccesing_machine'] = 'TOWER-REYLAB'

    if -1 != location.find('-RIP'):
        params['offset'] = 5
        if '-SS' not in location:
            params['beh_machine'] = '192.168.137.130'
            proccesing_machine_ips = ['192.168.137.226', '192.168.137.228']
            for ip in proccesing_machine_ips:
                ping_return = os.system(f'ping -c 1 -W 0.1 -q {ip}')
                if ping_return == 0:
                    params['proccesing_machine'] = ip
                    break
        params['system'] = 'RIP'
        params['online_notches'] = True
    elif -1 != location.find('-BRK'):
        params['offset'] = 5
        params['which_nsp_micro'] = 1
        if '-SS' not in location:
            params['beh_machine'] = '192.168.42.130'
            proccesing_machine_ips = ['192.168.42.226', '192.168.42.228']
            for ip in proccesing_machine_ips:
                ping_return = os.system(f'ping -c 1 -W 0.1 -q {ip}')
                if ping_return == 0:
                    params['proccesing_machine'] = ip
                    break
        params['online_notches'] = False
        params['system'] = 'BRK'
    else:
        raise ValueError('device inside MCW not found')

    return params

def BCM_location(location):
    params = {}
    params['ptb_priority_normal'] = 0
    params['ptb_priority_high'] = 1
    paths = ['C:\\Program Files\\Blackrock Microsystems\\NeuroPort Windows Suite',
             'C:\\Program Files (x86)\\Blackrock Microsystems\\Cerebus Windows Suite',
             'C:\\Program Files (x86)\\Blackrock Microsystems\\NeuroPort Windows Suite']
    for path in paths:
        if os.path.isdir(path):
            params['cbmex_path'] = os.pathsep.join([params.get('cbmex_path', ''), path])
            break
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 'SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Installer\\Folders') as key:
        for i in range(winreg.QueryInfoKey(key)[1]):
            valname = winreg.EnumValue(key, i)[0]
            if re.search('(Ripple\\\\Trellis\\\\Tools\\\\xippmex\\\\)$', valname):
                params['xippmex_path'] = valname
                break

    params['beh_machine'] = 1
    params['proccesing_machine'] = 2
    params['lang'] = 'english'
    params['device_resp'] = 'keyboard'
    params['keyboards'] = [0]
    params['remote_disk_root'] = 'Z:'
    params['use_photodiodo'] = True
    params['lang'] = 'english'
    params['device_resp'] = 'gamepad'
    params['which_nsp_comment'] = 1
    params['which_nsp_micro'] = 2
    params['acq_network'] = True
    params['additional_pics'] = 'pics_now'
    params['use_daq'] = True

    params['acq_remote_folder_in_beh'] = '/media/acq'
    params['with_acq_folder'] = True
    params['acq_is_beh'] = False
    params['acq_is_processing'] = False

    params['backup_in_processing'] = '/mnt/acq-hdd/'
    params['acq_remote_folder_in_processing'] = '/media/acq'
    params['with_acq_folder'] = True
    params['acq_is_processing'] = False

    params['pics_root_beh'] = 'C:\\Users\\EMU - Behavior\\Documents\\MATLAB\\Behavioral Tasks\\HR\\experimental_files\\pics'
    params['pics_root_processing'] = 'C:\\Users\\EMU - Behavior\\Documents\\MATLAB\\Behavioral Tasks\\HR\\experimental_files\\pics'

    params['trellis_data_path'] = 'C:\\Users\\emuca\\Trellis\\datafiles'

    params['central_data_path'] = ''
    params['root_beh'] = 'C:\\Users\\EMU - Behavior\\Documents\\MATLAB\\Behavioral Tasks\\HR\\experimental_files'
    params['root_processing'] = 'C:\\Users\\EMU - Behavior\\Documents\\MATLAB\\Behavioral Tasks\\HR\\experimental_files'

    params['mapfile'] = '*.map'
    params['processing_rec_metadata'] = 'C:\\Users\\EMU - Behavior\\Documents\\MATLAB\\Behavioral Tasks\\HR\\experimental_files\\rec_metadata'
    params['beh_rec_metadata'] = params['processing_rec_metadata']

    params['early_figures_full_path'] = 'C:\\Users\\EMU - Behavior\\Documents\\MATLAB\\Behavioral Tasks\\HR\\early_figures'

    if '-RIP' in location:
        params['system'] = 'RIP'
        params['use_BRK_comment'] = False
        params['online_notches'] = True
        params['additional_paths'] = ['C:\\Users\\EMU - Behavior\\Documents\\MATLAB\\Behavioral Tasks\\HR\\JoyMEX', params.get('xippmex_path', ''), params.get('cbmex_path', '')]
        params['ttl_device'] = 'MC'
    elif '-BRK' in location:
        params['use_BRK_comment'] = True
        params['online_notches'] = False
        params['system'] = 'BRK'
        params['ttl_device'] = 'LJ'
        params['additional_paths'] = ['C:\\Users\\EMU - Behavior\\Documents\\MATLAB\\Behavioral Tasks\\HR\\JoyMEX', params.get('cbmex_path', ''), 'C:\\Users\\EMU - Behavior\\Documents\\MATLAB\\Useful Codes\\For use with BlackRock']
    else:
        raise ValueError('System not found')
    return params