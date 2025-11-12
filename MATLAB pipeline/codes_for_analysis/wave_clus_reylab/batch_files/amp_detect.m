function [spikes,thr,index,remove_counter] = amp_detect(x, par)
% Detect spikes with amplitude thresholding. Uses median estimation.
% Detection is done with filters set by fmin_detect and fmax_detect. Spikes
% are stored for sorting using fmin_sort and fmax_sort. This trick can
% eliminate noise in the detection but keeps the spikes shapes for sorting.


sr = par.sr;
w_pre = par.w_pre;
w_post = par.w_post;

if isfield(par,'ref_ms')
    ref = floor(par.ref_ms * par.sr/1000);
else
    ref = par.ref; %for retrocompatibility
end

detect = par.detection;
stdmin = par.stdmin;
stdmax = par.stdmax;


if par.sort_order > 0
    xf = filt_signal(x,par.sort_order,par.sort_fmin,par.sort_fmax,par.sr,par);
else
    if par.preprocessing && ~isempty(par.process_info)
        xf = fast_filtfilt(par.process_info.SOS,par.process_info.G,x);
    else
        xf = x;
    end
end
if par.detect_order > 0
    xf_detect = filt_signal(x,par.detect_order,par.detect_fmin,par.detect_fmax,par.sr,par);
else
    if par.preprocessing && ~isempty(par.process_info)
       xf_detect = fast_filtfilt(par.process_info.SOS,par.process_info.G,x);
    else
        xf_detect = x;
    end    
end

%guarda(xf,xf_detect);

noise_std_detect = median(abs(xf_detect))/0.6745;
noise_std_sorted = median(abs(xf))/0.6745;
thr = stdmin * noise_std_detect;        %thr for detection is based on detect settings.
thrmax = stdmax * noise_std_sorted;     %thrmax for artifact removal is based on sorted settings.

index = [];
sample_ref = floor(ref/2);
% LOCATE SPIKE TIMES
switch detect
    case 'pos'
        nspk = 0;
        xaux = find(xf_detect(w_pre+2:end-w_post-2-sample_ref) > thr) +w_pre+1;
        xaux0 = 0;
        for i=1:length(xaux)
            if xaux(i) >= xaux0 + ref
                %[aux_unused, iaux] = max((xf(xaux(i):xaux(i)+sample_ref-1)));    %introduces alignment
                [pks,locs] = findpeaks(xf(xaux(i)-10:xaux(i)+sample_ref+10-1));
                if isempty(pks)
                    continue
                end
                [~,iM] = max(pks);                
                nspk = nspk + 1;
                %index(nspk) = iaux + xaux(i) -1;
                index(nspk) = minLoc(iM) + xaux(i) -10 -1;
                xaux0 = index(nspk);
            end
        end
    case 'neg'
        nspk = 0;
        xaux = find(xf_detect(w_pre+2:end-w_post-2-sample_ref) < -thr) +w_pre+1;
        xaux0 = 0;
        for i=1:length(xaux)
            if xaux(i) >= xaux0 + ref
                % [aux_unused, iaux] = min((xf(xaux(i):xaux(i)+sample_ref-1)));    %introduces alignment
                % [aux_unused, iaux] = min((xf(xaux(i)-10:xaux(i)+sample_ref+10-1)));    %introduces alignment
                [pks,locs] = findpeaks(-xf(xaux(i)-10:xaux(i)+sample_ref+10-1));
                if isempty(pks)
                    continue
                end
                [~,iM] = max(pks);                
                nspk = nspk + 1;
                % index(nspk) = iaux + xaux(i) -1;
                % index(nspk) = iaux + xaux(i) -10 -1;
                index(nspk) = locs(iM) + xaux(i) -10 -1;
                % if nspk==72
                %     figure
                %     plot(xf(index(nspk)-w_pre:index(nspk)+w_post));
                % end
                xaux0 = index(nspk);
            end
        end
    case 'both' % fixME need more info on spike set up
        % nspk = 0;
        % xaux = find(abs(xf_detect(w_pre+2:end-w_post-2-sample_ref)) > thr) +w_pre+1;
        % xaux0 = 0;
        % for i=1:length(xaux)
        %     if xaux(i) >= xaux0 + ref
        %        % [aux_unused, iaux] = max(abs(xf(xaux(i):xaux(i)+sample_ref-1)));    %introduces alignment
        % 
        %         nspk = nspk + 1;
        %         index(nspk) = iaux + xaux(i) -1;
        %         xaux0 = index(nspk);
        %     end
        % end
        nspk = 0;
        xaux = find(abs(xf_detect(w_pre+2:end-w_post-2-sample_ref)) > thr) +w_pre+1;
        xaux0 = 0;
        
        for i=1:length(xaux)
            if xaux(i) >= xaux0 + ref
                % Define the alignment window
                sig_window = xf(xaux(i)-10:xaux(i)+sample_ref+10-1);
                
                local_baseline = median(sig_window);
                
                sig_corrected = sig_window - local_baseline;
                
                [pks_pos, locs_pos] = findpeaks(sig_corrected);
                if ~isempty(pks_pos)
                    [max_p_true, idx_p] = max(pks_pos); % This is the true peak height
                    loc_p = locs_pos(idx_p);
                else
                    max_p_true = 0;
                    loc_p = 0;
                end
                
                [pks_neg, locs_neg] = findpeaks(-sig_corrected);
                if ~isempty(pks_neg)
                    [max_n_abs_true, idx_n] = max(pks_neg); % This is the true trough depth
                    loc_n = locs_neg(idx_n);
                else
                    max_n_abs_true = 0;
                    loc_n = 0;
                end
                
                if max_p_true >= max_n_abs_true 
                    best_loc = loc_p;
                elseif max_n_abs_true > max_p_true
                    best_loc = loc_n;
                else
                    continue
                end
                
                if best_loc > 0 
                    nspk = nspk + 1;
                    index(nspk) = best_loc + xaux(i) - 10 - 1;
                    xaux0 = index(nspk);
                end
            end
        end
end

% SPIKE STORING (with or without interpolation)
ls = w_pre+w_post;
spikes = zeros(nspk,ls+4);

xf(length(xf)+1:length(xf)+w_post)=0;
remove_counter = 0;
for i=1:nspk                          %Eliminates artifacts
    if max(abs( xf(index(i)-w_pre:index(i)+w_post) )) < thrmax
        spikes(i,:)=xf(index(i)-w_pre-1:index(i)+w_post+2);
    else
        remove_counter = 1 + remove_counter;
    end
end

aux = find(spikes(:,w_pre)==0);       %erases indexes that were artifacts
spikes(aux,:)=[];
index(aux)=[];

switch par.interpolation
    case 'n'
        spikes(:,end-1:end)=[];       %eliminates borders that were introduced for interpolation
        spikes(:,1:2)=[];
    case 'y'
        %Does interpolation
        spikes = int_spikes(spikes,par);
end
end

function filtered = filt_signal(x,order,fmin,fmax,sr,par)
    %HIGH-PASS FILTER OF THE DATA
    [b,a] = ellip(order,0.1,40,[fmin fmax]*2/sr);
    
    if par.preprocessing && ~isempty(par.process_info)
        [sos,g] = tf2sos(b,a);
        g = g * par.process_info.G;
        sos = [par.process_info.SOS; sos];
        filtered = fast_filtfilt(sos, g, x);
    else
        filtered = fast_filtfilt(b, a, x);      
    end
end
