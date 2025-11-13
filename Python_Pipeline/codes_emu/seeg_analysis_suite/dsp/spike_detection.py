# Author: Sunil Mathew
# Date: 10 Jan 2024
# Spike detection in micro channels, photodiode channel
import numpy as np
import scipy
from scipy import signal
from config import config
from joblib import Parallel, delayed

def detect_spikes(data, distance=45, det_neg=True):
    """
    Detects spikes in the data using the thresholding method.
    """
    thr_min, thr_max = compute_spike_threshold(data)
    if det_neg:
        spikes = signal.find_peaks(-data, height=thr_min, distance=distance)[0]
        # remove spikes that are above the artifact threshold
        spikes = spikes[np.where(-data[spikes] < thr_max)]
    else:
        # For photodiode channel
        thr_min *= 2
        spikes = signal.find_peaks(data, height=thr_min, distance=distance)[0]
    return spikes, thr_min

def smooth(a,WSZ):
    # https://stackoverflow.com/questions/40443020/matlabs-smooth-implementation-n-point-moving-average-in-numpy-python
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ    
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))

def detect_photo_spikes(data):
    """
    Detect photo diode spikes
    """
    thr = 2000
    sq_duration = 1500
    # Smoothen the signal
    # data = signal.savgol_filter(data, 51, 3)
    # spikes = signal.find_peaks(x=data, height=thr, distance=distance)[0]
    
    photo_smooth = smooth(data, 151)
    spikes = np.where(np.diff(photo_smooth > thr) == 1)[0]
    spikes = spikes - sq_duration
    # spike_widths = np.diff(spikes)
    # print(spike_widths[:10])
    spikes = spikes[::2] # Get only the start of the square wave
    # print(spikes[:10])
    return spikes, thr

def detect_spikes_in_segments(data, distance=45, photo=False):
    """
    Detects spikes in the data by splitting it into 5 min segments.
    """
    spikes = []
    thr_list = []
    # thr_running_avg = 0
    seg_sample_count = 5*60*30000 # 5 min segments
    multi_thread = True

    if multi_thread:
        if photo:
            spikes_thr = Parallel(n_jobs=-1)(delayed(detect_photo_spikes)(data[i:i+seg_sample_count]) for i in range(0, data.size, seg_sample_count))
        else:
            spikes_thr = Parallel(n_jobs=3)(delayed(detect_spikes)(data[i:i+seg_sample_count], distance) for i in range(0, data.size, seg_sample_count))
        seg_start_idxs = np.arange(0, data.size, seg_sample_count)
        for i, (s, thr) in enumerate(spikes_thr):
            seg_start_idx = seg_start_idxs[i]
            spikes.extend((s + seg_start_idx).astype(int).tolist())
            thr_list.append(thr)
    else:
        for i in range(0, data.size, seg_sample_count):
            if i + seg_sample_count > data.size:
                segment = data[i:]
            else:
                segment = data[i:i+seg_sample_count]
            if photo:
                segment_spikes, thr = detect_photo_spikes(segment)
            else:
                segment_spikes, thr = detect_spikes(segment, distance=distance)
            spikes.extend((segment_spikes + i).astype(int).tolist())
            # thr_running_avg += thr
            # thr_running_avg /= 2
            thr_list.append(thr)
    
    return spikes, thr_list

def compute_spike_threshold(data, std_min=5, std_max=50):
    """
    Computes the spike threshold.
    Steps:
    1. Calculate the median absolute deviation (MAD) of the data.
    2. Calculate the noise threshold as a multiple of the MAD.
    thr_min is used to detect spikes, thr_max is used to detect artifacts.
    """
    # Compute the noise
    noise = np.median(np.abs(data)) / 0.6745 # Median absolute deviation
    # Threshold to detect spikes
    thr_min = config['sEEG']['spike_params']['std_min'] * noise # std_min = 5
    # Threshold to detect artifacts
    thr_max = config['sEEG']['spike_params']['std_max'] * noise # std_max = 50
    return thr_min, thr_max

def extract_spikes(data, spikes, waveform_length=64):
    """
    Extracts spikes from the data.
    """
    spike_waveforms = []
    for t in spikes:
        waveform_start = max(0, t - 20) # 20 samples before the spike
        waveform_end = waveform_start + waveform_length
        waveform = data[waveform_start:waveform_end]
        if waveform.size < waveform_length:
            waveform = np.pad(waveform, (0, waveform_length - waveform.size), 'constant')
        waveform = waveform.astype(float).tolist()
        spike_waveforms.append(waveform)
    return spike_waveforms

def amp_detect_spikes_in_segments(xf, xf_detect):
    """
    Detects spikes in the data by splitting it into 5 min segments.
    """
    spike_idxs = []
    spike_waveforms = []
    thrs = []
    thr_running_avg = 0
    seg_sample_count = 5*60*30000 # 5 min segments

    for i in range(0, xf.size, seg_sample_count):
        if i + seg_sample_count > xf.size:
            xf_seg = xf[i:]
            xf_det_seg = xf_detect[i:]
        else:
            xf_seg = xf[i:i+seg_sample_count]
            xf_det_seg = xf_detect[i:i+seg_sample_count]

        spikes, thr, index, remove_counter = amp_detect(xf_seg, xf_det_seg)
        spike_waveforms.extend(spikes)
        spike_idxs.extend(index + i)
        thrs.append(thr)

        # thr_running_avg += thr
        # thr_running_avg /= 2

    # spike_idxs = np.array(spike_idxs)
    
    return spike_waveforms, spike_idxs, thrs

def amp_detect(xf, xf_detect):
    """
    Detect spikes with amplitude thresholding. Uses median estimation.
    Detection is done with filters set by fmin_detect and fmax_detect. Spikes
    are stored for sorting using fmin_sort and fmax_sort. This trick can
    eliminate noise in the detection but keeps the spikes shapes for sorting.
    """
    sr = 30000
    w_pre = 20
    w_post = 44
    stdmin = 5
    stdmax = 50
    ref = 45
    detect = 'neg'
    interpolation = 'n'

    noise_std_detect = np.median(np.abs(xf_detect)) / 0.6745
    noise_std_sorted = np.median(np.abs(xf)) / 0.6745
    thr = stdmin * noise_std_detect  # thr for detection is based on detect settings.
    thrmax = stdmax * noise_std_sorted  # thrmax for artifact removal is based on sorted settings.

    index = []
    sample_ref = ref // 2
    # LOCATE SPIKE TIMES
    nspk = 0
    if detect == 'neg':
        xaux = np.where(xf_detect[w_pre+1:-w_post-1-sample_ref] < -thr)[0] + w_pre + 1
        xaux0 = 0
        for i in range(len(xaux)):
            if xaux[i] >= xaux0 + ref:
                iaux = np.argmin(xf[xaux[i]:xaux[i]+sample_ref])
                nspk += 1
                index.append(iaux + xaux[i])
                xaux0 = index[nspk - 1]
    elif detect == 'pos':
        xaux = np.where(xf_detect[w_pre+2:-w_post-2-sample_ref] > thr)[0] + w_pre + 1
        xaux0 = 0
        for i in range(len(xaux)):
            if xaux[i] >= xaux0 + ref:
                iaux = np.argmax(xf[xaux[i]:xaux[i]+sample_ref])
                nspk += 1
                index.append(iaux + xaux[i])
                xaux0 = index[nspk - 1]
    elif detect == 'both':
        xaux = np.where(np.abs(xf_detect[w_pre+2:-w_post-2-sample_ref]) > thr)[0] + w_pre + 1
        xaux0 = 0
        for i in range(len(xaux)):
            if xaux[i] >= xaux0 + ref:
                iaux = np.argmax(np.abs(xf[xaux[i]:xaux[i]+sample_ref]))
                nspk += 1
                index.append(iaux + xaux[i])
                xaux0 = index[nspk - 1]

    # SPIKE STORING (with or without interpolation)
    ls = w_pre + w_post
    spikes = np.zeros((nspk, ls + 4))

    xf = np.append(xf, np.zeros(w_post))
    remove_counter = 0
    for i in range(nspk-1):
        if np.max(np.abs(xf[index[i]-w_pre:index[i]+w_post])) < thrmax:
            spikes[i] = xf[index[i]-w_pre-1:index[i]+w_post+3]
        else:
            remove_counter += 1

    aux = np.where(spikes[:, w_pre] == 0)[0]
    spikes = np.delete(spikes, aux, axis=0)
    index = np.delete(index, aux)

    if interpolation == 'n':
        spikes = np.delete(spikes, [ls, ls+1], axis=1)
        spikes = np.delete(spikes, [0, 1], axis=1)
    elif interpolation == 'y':
        # Does interpolation
        spikes = int_spikes(spikes)

    index = index.astype(int)

    return spikes, thr, index, remove_counter

def int_spikes(spikes, w_pre=20, w_post=44):
    """
    Interpolates with cubic splines to improve alignment.
    """
    ls = w_pre + w_post
    detect = 'neg'
    int_factor = 5
    nspk = spikes.shape[0]
    extra = (spikes.shape[1] - ls) // 2

    s = np.arange(spikes.shape[1])
    ints = np.arange(1/int_factor, spikes.shape[1]-1, 1/int_factor)
    ints = np.append(ints, spikes.shape[1]-1)
    spikes1 = np.zeros((nspk, ls))
    if nspk > 0:
        intspikes = scipy.interpolate.interp1d(s, spikes, axis=1)(ints)
        if detect == 'neg':
            maxi = np.min(intspikes[:, (w_pre+extra-1)*int_factor-1:(w_pre+extra+1)*int_factor-1], axis=1)
        elif detect == 'pos':
            maxi = np.max(intspikes[:, (w_pre+extra-1)*int_factor:(w_pre+extra+1)*int_factor], axis=1)
        elif detect == 'both':
            maxi = np.max(np.abs(intspikes[:, (w_pre+extra-1)*int_factor:(w_pre+extra+1)*int_factor]), axis=1)
        iaux = np.argmin(intspikes[:, (w_pre+extra-1)*int_factor:(w_pre+extra+1)*int_factor], axis=1)
        iaux += (w_pre+extra-1)*int_factor

        for i in range(nspk):
            spikes1[i] = intspikes[i, iaux[i]-w_pre*int_factor+int_factor:iaux[i]+w_post*int_factor:int_factor]

    return spikes1