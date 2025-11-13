import os
import numpy as np
import glob
from core.parameter_functions.Parameters import par
from scipy.io import loadmat
import sys ; sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/codes_emu/codes_for_analysis/new_processing_pipeline')
from parse_ripple import parse_ripple
from core.filtering import filtering
from core.plot_continuous_bundles import PlotBundles
from core.spikeDetection import Spikes
from core.Collision import Collision
from core.Clustering import clustering
import multiprocessing


param = par()
param.parallel = True
if param.micros:
    binary_ext = '.NC5'
    ext = '.ns5'
else:
    binary_ext = '.NC3'
    ext = '.nf3'

file_paths = glob.glob("input/*"+binary_ext)
if file_paths == []:
    file_paths_ns5 = glob.glob("input/*"+ext)
    parse_ripple(file_paths_ns5)
    file_paths = glob.glob("input/*"+binary_ext)
NSx_file_path = os.path.abspath(glob.glob("input/NSx.mat")[0])

#ncx = NCxFiles(NSx_file=NSx_file_path)
#power_data, raw_data = ncx.read_channels(tmin=0,selected_chan_IDs=[257, 258, 259])
dir_path = os.path.dirname(os.path.realpath(__file__))
pics_used_dir = dir_path + '/input/pics_used'

metadata = loadmat(NSx_file_path)
nsx = metadata['NSx']
if param.micros:
    channels = nsx['chan_ID'][0][list(set(np.where(nsx['unit']=='uV')[1]) & set(np.where(nsx['sr']==30000)[1]))]
else:
    channels = nsx['chan_ID'][0][list(set(np.where(nsx['sr']==2000)[1]))]

#filtering
filter = filtering(save_fig=False,show_img=True,direc_resus_bae=os.path.dirname(os.path.realpath(__file__)),
                    resus_folder_name='spectra',direc_raw=os.path.dirname(os.path.realpath(__file__)),with_NoNotch = False,
                    time_plot_duration = 1,freq_line=60,parallel=param.parallel,k_periodograms=200,notch_filter=True,spectrum_resolution=0.5)
filter.new_check_lfp_power_NSX(metadata, channels)

#plot bundles
#if param.parallel:
#    num_workers = os.cpu_count()*2
#    pool = QThreadPool()
#    pool.setMaxThreadCount(num_workers)
#    plt = PlotBundles()
#    worker = TW.WorkerObject(plt.plot,
#                                     parent=None,
#                                     s='plot_bundles',par = param,nsx_file=nsx,notchfilter=1)
#    pool.start(worker)
#else:
#    plt = PlotBundles()
#    plt.plot(nsx_file = nsx, par = param,notchfilter=1)
plt = PlotBundles()
plt.plot(nsx_file = nsx, par = param,notchfilter=1)

#spike detection
if param.micros:
    ch_temp = []
    if param.fast_analysis or param.nowait:
        neg_thr_channels = channels
        pos_thr_channels = []
    else:
        ch_temp = input(f'Currently, Channels = {channels}. \nIf you want to keep it like that, press enter.\nOtherwise, enter the new vector and press enter ')
    if ch_temp !='':
        channels = ch_temp
    
    neg_thr_channels = input('Enter the vector with neg_thr_channels and press enter. Press enter to use all channels ')
    if ch_temp == '':
        neg_thr_channels = channels
        pos_thr_channels = np.array([])
    else:
        pos_thr_channels = input('Enter the vector with pos_thr_channels and press enter. Press enter for empty array ')

    #start parallel process
    del param
    param = par()
    param.detection = 'neg'
    param.sr = 30000
    param.detect_fmin = 300
    param.detect_fmax = 3000
    param.auto = 0
    param.mVmin = 50
    param.w_pre=20                       
    param.w_post=44                     
    param.min_ref_per=1.5                                    
    param.ref = np.floor(param.min_ref_per*param.sr/1000)                  
    param.ref = param.ref
    param.factor_thr=5
    param.detect_order = 4
    param.sort_order = 2
    param.detect_fmin = 300
    param.sort_fmin = 300
    param.stdmin = 5
    param.stdmax = 50
    param.ref_ms = 1.5
    param.preprocessing = True
    param.minus_one = 0

    print('starting spike detection')
    param.detection = 'neg'
    if param.parallel:
        spike = Spikes(par=param,nsx=nsx)
        if neg_thr_channels.size:
            with multiprocessing.Pool(processes=10) as pool:
                pool.imap(spike.get_spikes,neg_thr_channels,chunksize=10)#might need to use imap()
                pool.close()
                pool.join()

        param.detection = 'pos'
        if pos_thr_channels.size:
            with multiprocessing.Pool(processes=10) as pool:
                pool.imap(spike.get_spikes,pos_thr_channels,chunksize=10)#might need to use imap()
                pool.close()
                pool.join()
                

        param.detection ='both'
        both_thr_channels = np.setdiff1d(np.setdiff1d(channels,neg_thr_channels),pos_thr_channels)
        if both_thr_channels.size:
            with multiprocessing.Pool(processes=10) as pool:
                pool.imap(spike.get_spikes,both_thr_channels,chunksize=10)#might need to use imap()
                pool.close()
                pool.join()
                
                
        print('spike detection done')
        
    else:
        if neg_thr_channels.size:
            spike = Spikes(par=param,nsx=nsx)
            for channel in neg_thr_channels:
                spike.get_spikes(channel=channel[0][0])

        param.detection = 'pos'
        if pos_thr_channels.size:
            for channel in pos_thr_channels:
                spike.get_spikes(channel=channel[0][0])
        param.detection ='both'
        both_thr_channels = np.setdiff1d(np.setdiff1d(channels,neg_thr_channels),pos_thr_channels)
        if not both_thr_channels.size:
            for channel in both_thr_channels:
                spike.get_spikes(channel[0][0])
        print('spike detection done')

    col = Collision(channels,nsx)
    col.separate_collisions()
    clus = clustering()
    clus.do_clustering(par,nsx)