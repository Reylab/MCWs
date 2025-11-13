function  calculate_notches(channels,pxx_db,fs,db_thr,folder)
    par = struct;
    par.span_smooth = 21;
    par.db_thr = db_thr;
%     par.min_freq = 300;
    par.min_freq = 150;
    par.sr = 30000;
    %par.max_freq = 3000;
    par.max_freq = 8000;
    first_freq = find(fs<par.min_freq,1,'last')+1-par.span_smooth;
    inds_freqs_search_notch = first_freq:(find(fs>=par.max_freq,1,'first')-1+par.span_smooth);
    fs = fs(inds_freqs_search_notch);
    nchannels = length(channels);
    process_info_out = cell(nchannels,1);
    for i=1:nchannels
       process_info_out{i} = compute_notches_struct(channels(i),...
           pxx_db{i}(inds_freqs_search_notch),fs,par);
    end
    
    if exist(fullfile(folder,'pre_processing_info.mat'),'file')
        load(fullfile(folder,'pre_processing_info.mat'),'process_info')
        process_temp = cell2mat(process_info_out(~cellfun('isempty',process_info_out)));
        for j=1:length(process_temp)
            if isempty(process_info)
                ind_preprocess = [];
            else
                ind_preprocess = find([process_info(:).chID]==process_temp(j).chID);
            end
            if isempty(ind_preprocess)
                ind_preprocess = length(process_info)+1;
            end
            process_info(ind_preprocess).chID = process_temp(j).chID;
            process_info(ind_preprocess).notches =process_temp(j).notches;
            process_info(ind_preprocess).n_notches = process_temp(j).n_notches;
        end
        save(fullfile(folder,'pre_processing_info.mat'),'process_info','inds_freqs_search_notch','-append')
    else
        process_info = cell2mat(process_info_out(~cellfun('isempty',process_info_out)));
        save(fullfile(folder,'pre_processing_info.mat'),'process_info','inds_freqs_search_notch')
    end
    
end

function info = compute_notches_struct(channel,pxx_db,fs,par)
    pxx_thr_db = movmedian(pxx_db,par.span_smooth) + par.db_thr;

    i3k = find(fs>=2950,1);
    pxx_thr_db(i3k:end) = pxx_thr_db(i3k);
    used_notches = [];
    abs_db = [];
    diff_db = [];
    
    supra_thr = find(pxx_db > pxx_thr_db);    
    if ~isempty(supra_thr)
        max_amp4notch = max(pxx_db- pxx_thr_db);
        temp_supra=find(diff(supra_thr)>1);
        inds_to_explore=[supra_thr(1); supra_thr(temp_supra+1)];
        if isempty(temp_supra)
            sample_above = numel(supra_thr);
        else
            sample_above = [temp_supra(1); diff(temp_supra); numel(supra_thr)-find(supra_thr==inds_to_explore(end))+1];
        end
        
        notch_idx = find(sample_above(1:length(inds_to_explore))>1); %only uses sample_above(jj)>1
        used_notches = zeros(length(notch_idx),1);
        bw_notches = zeros(length(notch_idx),1);
        abs_db = zeros(length(notch_idx),1);
        diff_db = zeros(length(notch_idx),1);
        
        for i = 1:length(notch_idx)
            jj = notch_idx(i);
            [~, iaux] = max(pxx_db(inds_to_explore(jj):inds_to_explore(jj)+sample_above(jj)-1));    %introduces alignment
            centre_sample = mean(inds_to_explore(jj):inds_to_explore(jj)+sample_above(jj)-1);
            ind_max = iaux + inds_to_explore(jj) - 1;
            if mod(centre_sample,1)==0.5 && (pxx_db(floor(centre_sample))> pxx_db(ceil(centre_sample)))
                centre_sample = floor(centre_sample);
            else
                centre_sample = ceil(centre_sample);
            end
            amp4notch = pxx_db(ind_max)-pxx_thr_db(ind_max);
            used_notches(i) = fs(centre_sample);
            bw_notches(i) = (fs(2)-fs(1))*sample_above(jj)*2*amp4notch/max_amp4notch;
            abs_db(i) = pxx_db(ind_max);
            diff_db(i) = amp4notch;
        end   
    end
    
    
    
    info=struct();
    info.chID = channel;
    info.notches.Z = [];
    info.notches.P = [];
    info.notches.K = zeros(numel(used_notches),1);
    info.notches.freq = used_notches;
    info.notches.abs_db = abs_db;
    info.notches.diff_db = diff_db;
    info.notches.thr_db = pxx_thr_db;
    info.n_notches = numel(used_notches);
    if numel(used_notches)>0
        for i= 1:numel(used_notches)
            nfi = used_notches(i);
            w = nfi/(par.sr/2);
            if nfi<295
                max_bw = 3;
            else
                max_bw = 5;
            end
            bw_notches(i) = min(max(bw_notches(i),1),max_bw);
            bw = bw_notches(i)/(par.sr/2);
    
            [b_notch,a_notch] = iirnotch(w,bw);
            [zi,pi,ki] = tf2zpk(b_notch,a_notch);
            info.notches.Z(end+1:end+2) = zi;
            info.notches.P(end+1:end+2) = pi;
            info.notches.K(i) = ki;
        end
        info.notches.bw_notches = bw_notches;
    else
        info.notches.bw_notches = [];
    end
end