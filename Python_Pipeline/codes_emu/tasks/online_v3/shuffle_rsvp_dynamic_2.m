function [experiment, scr_config, extra_rep_pics] = shuffle_rsvp_dynamic_2(experiment, pics2use, scr_id)
    % Author: Sunil Mathew
    % based on shuffle_rsvpSCR_online3.m
    % Creates sequences to show pic2use as many repetitions as configured
    % In last block, checks how many times a pic is shown and creates sequences to ensure that
    % each pics gets shown max_trials times or max_trials+1 to have uniform
    % seq_length
    extra_rep_pics = [];
    if numel(pics2use) == 0
        scr_config = struct;
        scr_config.ISI=0.5;
        scr_config.seq_length=0;
        scr_config.Nseq=0;
        scr_config.Nrep=0;
        return
    end

       
    
    % 30 secs is the minimum duration per trial (the actual length will be between min_seq_length aNrepxseqnd 2*min_seq_length)
    ISI = 0.5;
    min_seq_length = ceil(30/ISI);     
    ImageNames = experiment.ImageNames(pics2use,:);
    
    all_pics_list = [];
    if scr_id < experiment.N_BLKS
        n_rep = experiment.NREP(scr_id);
        n_pics = numel(pics2use);
        if n_pics < min_seq_length
            % we'll repeat the same pic twice in the sequence.
            seq_length = n_pics * 2;
            n_seq = ceil(n_rep / 2);
            n_rep = n_rep + mod(n_rep, 2);
        else
            Nseqxrep = floor(n_pics/min_seq_length); % how many seqs per rep
            seq_length = ceil(n_pics/Nseqxrep);
            n_seq = Nseqxrep * n_rep;
        end        
        
        while n_rep > 0
            all_pics_list = [all_pics_list pics2use(randperm(length(pics2use)))];
            n_rep = n_rep - 1;
        end
    else
        % last block sequences creation
        n_rep = experiment.MAX_TRIALS;
        stim_trial_count_arr = unique(ImageNames.stim_trial_count);
        stim_trial_count_arr = sort(stim_trial_count_arr);
        pics_groups = {};
        for idx=1:length(stim_trial_count_arr)
            stim_trial_count = stim_trial_count_arr(idx);
            req_stim_trial_count = n_rep - stim_trial_count;
            pics = pics2use(ImageNames.stim_trial_count == stim_trial_count);
            pics_groups{end+1} = struct;
            pics_groups{end}.pics = pics;
            pics_groups{end}.req_stim_trial_count = req_stim_trial_count;
        end
        
        while pics_groups{1}.req_stim_trial_count > 0
            pics_list = [];
            for pics_group_idx=1:numel(pics_groups)
                pics_group = pics_groups{pics_group_idx};
                if pics_group.req_stim_trial_count > 0
                    pics = pics_group.pics;
                    pics_list = [pics_list pics(randperm(length(pics)))];
                    pics_groups{pics_group_idx}.req_stim_trial_count = pics_groups{pics_group_idx}.req_stim_trial_count - 1;
                end
            end
            all_pics_list = [all_pics_list pics_list(randperm(length(pics_list)))];
        end
        Nseqxrep = floor(numel(all_pics_list)/min_seq_length); % how many seqs per rep
        seq_length = ceil(numel(all_pics_list)/Nseqxrep);
    end    
    
    
    if contains(experiment.subtask,'DynamicSeman') || ...
       contains(experiment.subtask,'CategLocaliz')
        colors = [[0 0 0];[0 0 0]];
    else 
        colors = [[255 0 0];[255 255 0]];
    end
    
    if numel(ISI)~=1
        error('This code is meant to be used with a single ISI value\n')
    end    
    
    [experiment, sequences, extra_rep_pics] = create_sequences(experiment, pics2use, all_pics_list, seq_length, n_rep);
    sequences = cell2mat(sequences);
    n_seq = size(sequences, 2);
    order_pic = reshape(sequences, [seq_length, 1, n_seq]);
    
    ISI = sort(ISI);
    seq_length = size(order_pic,1);
    n_seq = size(order_pic,3);
    
    tmean_ch = 10;
    nmean_ch = round(seq_length*ISI/tmean_ch);
    
    order_ISI = ones(1,n_seq);
    
    col_vals = [[1 1];[1 2];[2 1];[2 2]];
    
    %pics change between ISI/4 and 3*ISI/4
    
    nchanges_blank=zeros(1,n_seq); % no changes in the blank
    nchanges_pic=NaN*ones(1,n_seq);
    lines_change = cell(1,n_seq);
    % subcell 1 has another subcell where each row is a change. 
    % col 1 is the blank number where it changes
    % col 2 the time when it changes (in sec from the onset (previous times))
    % col 3 is the color of top line
    % col 4 is the color of bottom line
    
    % subcell 2 has another subcell where each row is a change. 
    % col 1 is the ISI (1 being the shortest)
    % col 2 the time when it changes (in sec from the onset (previous times))
    % col 3 is the color of top line
    % col 4 is the color of bottom line
    % col 5 is the pic in the sequence where the change takes place
    
    for irep=1:n_seq
        for k=1:4
            lines_change{irep}{1}{nchanges_blank(irep)+1,k}=NaN;
        end
    
        nchanges_pic(irep) = nmean_ch+randi(3)-2;
        t2=randperm(seq_length-2);
        t2=sort(t2(1:nchanges_pic(irep)))+1;
        for j=1:nchanges_pic(irep)
            lines_change{irep}{2}{j,1} = 1;
            lines_change{irep}{2}{j,2} = rand(1)*ISI(1)/2 + ISI(1)/4;
            lines_change{irep}{2}{j,5} = t2(j);
            lines_change{irep}{2}{j,6} = 1;
        end    
        for k=1:6
            lines_change{irep}{2}{nchanges_pic(irep)+1,k}=NaN;
        end
    end
    
    for irep=1:n_seq
        ok=0;
        while ok==0
            cols_seq=[];
            for gg=1:ceil(max(nchanges_pic)/4)+1
                cols_seq = [cols_seq randperm(4)];
            end
            if ~any(diff(cols_seq)==0)
                ok=1;
            end
        end
        color_start.up{irep} = colors(col_vals(cols_seq(1),1),:);
        color_start.down{irep} = colors(col_vals(cols_seq(1),2),:);
        for kk=1:size(lines_change{irep}{2},1)-1
            lines_change{irep}{2}{kk,3}=colors(col_vals(cols_seq(kk+1),1),:);
            lines_change{irep}{2}{kk,4}=colors(col_vals(cols_seq(kk+1),2),:);
        end
    end
    
    scr_config = struct;
    scr_config.order_pic = order_pic;
    scr_config.ISI=ISI;
    scr_config.seq_length=seq_length;
    scr_config.Nseq=n_seq;
    scr_config.Nrep=n_rep;
    scr_config.order_ISI=order_ISI;
    scr_config.color_start=color_start;
    scr_config.nchanges_blank=nchanges_blank;
    scr_config.nchanges_pic=nchanges_pic;
    scr_config.lines_change=lines_change;
end

function [experiment, sequences, extra_rep_pics] = create_sequences(experiment, ...
                                                    pics2use, pics_list, ...
                                                    seq_length, n_rep)
    sequences = {};  
    extra_rep_pics = [];
    
    for i = 1:seq_length:length(pics_list)
        seq = pics_list(i:min(i+seq_length-1, end));
        sequences{end+1} = seq;    
    end
    
    experiment.ImageNames.stim_trial_count(pics2use) = n_rep;
    
    if length(sequences{end}) < seq_length
        pics_reqd_count = seq_length - length(sequences{end});
        extra_rep_pics = setdiff(pics2use, sequences{end});
        extra_rep_pics = extra_rep_pics(1:pics_reqd_count);
        sequences{end} = [sequences{end} extra_rep_pics];
        experiment.ImageNames.stim_trial_count(extra_rep_pics) = ...
        experiment.ImageNames.stim_trial_count(extra_rep_pics) + 1;
    end
    
    % convert to indices
    for seq_idx=1:numel(sequences)
        seq = sequences{seq_idx};
        [~, seq] = ismember(seq, pics2use);
        sequences{seq_idx} = seq';
    end
    [~, extra_rep_pics] = ismember(extra_rep_pics, pics2use);
end
