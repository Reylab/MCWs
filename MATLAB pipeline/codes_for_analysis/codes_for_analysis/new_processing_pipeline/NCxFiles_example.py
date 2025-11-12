from NCxFiles import NCxFiles
import glob

nsx_file = '/mnt/data0/sEEG_DATA/MCW-FH_010/EMU/EMU-002_subj-MCW-FH_010_task-RSVPDynamicScr_run-01/NSx.mat'
ncx = NCxFiles(NSx_file=nsx_file)
power_dict,raw_dict,units = ncx.read_channels(0,60,['257'])
power_dict2,raw_dict,units = ncx.read_channels(0,60,['mLTP01 raw'])

power_dict,raw_dict,units = ncx.read_bundles(0,60,['mLTP01'])

