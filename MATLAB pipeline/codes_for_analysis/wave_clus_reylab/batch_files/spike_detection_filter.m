function xf_detect = spike_detection_filter(x, par)
%this function filter the signal, using the detection filter. Is used in the
%readInData class. 

sr = par.sr;
fmin_detect = par.detect_fmin;
fmax_detect = par.detect_fmax;


% HIGH-PASS FILTER OF THE DATA
if par.detect_order>0
    [b,a] = ellip(par.detect_order,0.1,40,[fmin_detect fmax_detect]*2/sr);
    
    if par.preprocessing && ~isempty( par.process_info)
        [sos,g] = tf2sos(b,a);
        g = g * par.process_info.G;
        sos = [par.process_info.SOS; sos];
        xf_detect = fast_filtfilt(sos, g, x);
    else
        xf_detect = fast_filtfilt(b, a, x); 
    end
    
else
    if par.preprocessing && ~isempty( par.process_info)
        x = pre_processing(x,par.filename);
    end
    xf_detect = x;  
end