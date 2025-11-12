function separate_task_spikes(channels, use_blanks)
separate_task_spks_tic = tic;
seq_end_offset = 3000; % 3 seconds after last pic onset
seq_beg_end_mat = get_seq_beg_end_mat(use_blanks, seq_end_offset);

load('NSx','NSx');
NSx = NSx(ismember(cell2mat({NSx.chan_ID}),channels));
parfor k= 1:length(channels)
    update_ch_valid_spktimes(NSx(k).output_name, seq_beg_end_mat, seq_end_offset);
end

separate_task_spks_toc = toc(separate_task_spks_tic);
fprintf("separate_task_spikes DONE in %s seconds. \n", num2str(separate_task_spks_toc, '%2.2f'));

end

function valid_spktimes = get_valid_spktimes(spktimes, seq_beg_time, seq_end_time)
    valid_spktimes = spktimes(spktimes>=seq_beg_time & spktimes<=seq_end_time);
end

function update_ch_valid_spktimes(ch_label, seq_beg_end_mat, seq_end_offset)
    ch_spks_file = sprintf('%s_spikes.mat', ch_label);
    SPK = load(ch_spks_file); 
    par = SPK.par;
    par.seq_end_offset = seq_end_offset;
    ch_valid_spktimes = [];  
    
    for seq=1:size(seq_beg_end_mat,1)
        seq_beg_time = seq_beg_end_mat(seq,1);
        seq_end_time = seq_beg_end_mat(seq,2);
        valid_spktimes = get_valid_spktimes(SPK.index_all, seq_beg_time, seq_end_time);
        ch_valid_spktimes = [ch_valid_spktimes; valid_spktimes'];
    end
    
    mask_taskspks = ismember(SPK.index_all, ch_valid_spktimes);
    mask_validspks = SPK.mask_nonart & mask_taskspks;
    index = SPK.index_all(mask_validspks);
    spikes = SPK.spikes_all(mask_validspks,:);
    fprintf("%2.2f%%(%d/%d) valid task spikes in %s\n", numel(index)/numel(SPK.index_all)*100, ...
            numel(index), numel(SPK.index_all), ch_spks_file);
    save(ch_spks_file, 'par', 'index', 'spikes', 'mask_taskspks', ...
                       'seq_beg_end_mat', "-append")
end

function seq_beg_end_mat = get_seq_beg_end_mat(use_blanks, seq_end_offset)
    load('experiment_properties_online3.mat', 'experiment', ...
     'scr_config_cell', 'scr_end_cell')
    load finalevents;
    if ~exist("seq_end_blanks_cell","var") && use_blanks
        use_blanks = false;
    end
    if ~exist("seq_beg_blanks_cell","var") || ~exist("trial_on_times","var")
        error(['finalevents does not have seq_beg_blanks_cell or trial_on_times. ' ...
            'run extract_blank_on_events_ripple'])
    end

    Nscr = numel(scr_config_cell);
    n_scr_ended = numel(scr_end_cell);

    if Nscr>n_scr_ended
        warning('using just the completed screenings (%d/%d)', Nscr, n_scr_ended);
        Nscr = n_scr_ended;
    end
    seq_ctr = 1;    
    seq_beg_end_cell = cell(Nscr, 1);
    for scri = 1:Nscr
        subscr_seq_beg = seq_beg_blanks_cell{scri};
        seq_count = numel(subscr_seq_beg);
        subscr_seq_beg_end = cell(seq_count, 1);
        if use_blanks
            subscr_seq_end = seq_end_blanks_cell{scri};
            for seq = 1:seq_count
                seq_beg_time = subscr_seq_beg{seq}(1);
                seq_end_time = subscr_seq_end{seq}(2);
                subscr_seq_beg_end{seq} = [seq_beg_time seq_end_time];
            end
        else  % For patients before MCW-FH-016
            subscr_pics_onset = pics_onset{scri};
            seq_count = size(subscr_pics_onset, 3);
            for seq = 1:seq_count
                seq_beg_time = subscr_seq_beg{seq}(1);
                seq_pics_onset = subscr_pics_onset(:,:,seq);
                seq_end_time = seq_pics_onset(end) + seq_end_offset;
                trial_on_time = trial_on_times(seq_ctr);
                if seq_end_time > trial_on_time
                    seq_end_time = trial_on_time;
                end
                subscr_seq_beg_end{seq} = [seq_beg_time seq_end_time];
                seq_ctr = seq_ctr + 1;
            end
        end
        seq_beg_end_cell{scri} = subscr_seq_beg_end;
    end
    seq_beg_end_mat = cell2mat(cellfun(@(x) cell2mat(x), seq_beg_end_cell, 'UniformOutput', false));
end
