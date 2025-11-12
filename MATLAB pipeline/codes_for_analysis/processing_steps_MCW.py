import os
import sys
import glob
sys.path.append(os.path.join(os.path.dirname(__file__), 'new_processing_pipeline'))

from parse_ripple import *


def processing_steps_MCW(which_system_micro='RIP',fast_analysis=0,nowait=True,micros=True,
                         remove_ref_chs=[],do_power_plot=1,notchfilter=1,do_sorting=1,
                         do_loop_plot=1,extract_events=1,extra_stims_win=0):
    
    global filenames
    
    if micros:
        ftype = 'ns5'
    else:
        if which_system_micro == 'RIP':
            ftype = 'nf3'
        elif which_system_micro == 'BRK':
            ftype = 'ns3'

    if which_system_micro == 'BRK':
        fname_prefix = '*.'
    elif which_system_micro == 'RIP':
        fname_prefix = '*_RIP.'
    else:
        fname_prefix = ''

    if 'filenames' not in globals():
        A = glob.glob(fname_prefix + ftype)
        if not A:
            raise ValueError(f'There are no {ftype} files in this folder')
        else:
            filenames = A
    else:
        print(f'variable filenames already exists and is equal to {filenames}')

    parse_ripple(filenames=filenames, overwrite=True)


if __name__ == '__main__':
    file_path = os.path.join(os.getcwd(), 'ns_Data', 'EMU-001_subj-MCW-FH_test_task-gaps\EMU-001_subj-MCW-FH_test_task-gaps_run-01_RIP.ns5')
    filenames = [file_path]
    processing_steps_MCW()

