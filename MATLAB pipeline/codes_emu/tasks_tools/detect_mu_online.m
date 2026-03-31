function index = detect_mu_online(data,ch_filter, par,detect)
sr = par.sr;
w_pre = par.w_pre;
w_post = par.w_post;
stdmin = par.stdmin;
stdmax = par.stdmax;
ref = floor(par.ref_ms*sr/1000);
if ~isa(data,'double') || ~isa(data,'single') 
    xf_detect = fast_filtfilt(ch_filter{1}, ch_filter{2}, single(data)); 
else
    xf_detect = fast_filtfilt(ch_filter{1}, ch_filter{2}, data);      
end
noise_std_detect = fastAbsMedian(xf_detect)/0.6745;
thr = stdmin * noise_std_detect;        %thr for detection is based on detect settings.
thrmax = stdmax * noise_std_detect;     %thrmax for artifact removal is based on sorted settings.

index = [];

% LOCATE SPIKE TIMES
switch detect
    case 1
        xaux = find(xf_detect(w_pre+2:end-w_post-2-floor(ref/2)) > thr) +w_pre+1;
        xaux0 = 0;
        for i=1:length(xaux)
            if xaux(i) >= xaux0 + ref
                aux_sig = xf_detect(xaux(i):xaux(i)+floor(ref/2)-1);
                [~, iaux]=max(aux_sig);    %introduces alignment
                if iaux == 1 && all(aux_sig(1) > aux_sig(2:end))
                    continue
                end
                index_aux = iaux + xaux(i) -1;
                if max(abs( xf_detect(index_aux-w_pre:index_aux+w_post) )) > thrmax
                    continue
                end
                index(end+1) = index_aux;
                xaux0 = index_aux;
            end
        end
    case -1
        xaux = find(xf_detect(w_pre+2:end-w_post-2-floor(ref/2)) < -thr) +w_pre+1;
        xaux0 = 0;
        for i=1:length(xaux)
            if xaux(i) >= xaux0 + ref
                aux_sig = xf_detect(xaux(i):xaux(i)+floor(ref/2)-1);
                [~, iaux]=min(aux_sig);    %introduces alignment
                if iaux == 1 && all(aux_sig(1) < aux_sig(2:end))
                    continue
                end
                index_aux = iaux + xaux(i) -1;
                if max(abs( xf_detect(index_aux-w_pre:index_aux+w_post) )) > thrmax
                    continue
                end
                index(end+1) = index_aux;
                xaux0 = index_aux;
            end
        end
    case 0
        xaux = find(abs(xf_detect(w_pre+2:end-w_post-2-floor(ref/2))) > thr) +w_pre+1;
        xaux0 = 0;
        for i=1:length(xaux)
            if xaux(i) >= xaux0 + ref
                aux_sig = abs(xf_detect(xaux(i):xaux(i)+floor(ref/2)-1));
                [~, iaux] = max(aux_sig);    %introduces alignment
                if iaux == 1 && all(aux_sig(1) > aux_sig(2:end))
                    continue
                end
                index_aux = iaux + xaux(i) -1;
                if max(abs( xf_detect(index_aux-w_pre:index_aux+w_post) )) > thrmax
                    continue
                end
                index(end+1) = index_aux;
                xaux0 = index_aux;
            end
        end
end


end