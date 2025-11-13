import scipy.signal as signal
import numpy as np


def filt_signal(x,order,fmin,fmax,sr,par,s):
    b,a = signal.ellip(N=order,rp=0.1,rs=40,btype='bandpass',Wn=np.array([fmin,fmax])*2/sr)
    if par['preprocessing'] and par['process_info']:
        sos = signal.tf2sos(b,a)
        try:
            sos = np.concatenate((sos,s),axis=0)   
        except:
            sos = sos
        filtered = signal.sosfiltfilt(sos,x,padlen = 3*(s.shape[0]))#might need to add the padlen3*(2*len(sos))
    else:
        filtered = signal.filtfilt(b,a,x)#might need to add the padlen
    return filtered.tolist()
x = filt_signal(x_raw,detect_order,detect_fmin,detect_fmax,sr,param,s)