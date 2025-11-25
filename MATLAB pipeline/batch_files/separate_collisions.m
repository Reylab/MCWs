function separate_collisions(channels)
separate_collisions_tic = tic;
load('NSx','NSx');
NSx = NSx(ismember(cell2mat({NSx.chan_ID}),channels));
t_win = 0.5;
bundle_min_art = 6;

bundles_to_explore = unique({NSx.bundle});

% for ibun = 1:length(bundles_to_explore)
for ibun = 1:length(bundles_to_explore)
    bundles_collisions_tic = tic;
    pos_chans_probe = find(arrayfun(@(x) (strcmp(x.bundle,bundles_to_explore{ibun})),NSx));
    posch = pos_chans_probe(1);
    if ~NSx(posch).is_micro
        continue
    end

    % fprintf('%s ,', NSx(posch).bundle)
    all_spktimes = [];
    which_chan = [];
    inds = struct;
%     SPK=struct;
    for k= 1:length(pos_chans_probe)
        SPK = load(sprintf('%s_spikes.mat',NSx(pos_chans_probe(k)).output_name));
        if isfield(SPK,'index_all')
            inds(k).spktimes = SPK.index_all;          
        else
            inds(k).spktimes = SPK.index;           
        end
        all_spktimes = [all_spktimes inds(k).spktimes];
        which_chan = [which_chan NSx(pos_chans_probe(k)).chan_ID*ones(size(inds(k).spktimes))];
    end
    [all_spktimes,II] = sort(all_spktimes);
    which_chan = which_chan(II);
    is_artifact = false(size(all_spktimes));
    artifact_idxs = [];

    % figure
    % all_times = inds(2).spktimes;
    % all_times_plot = all_times(all_times>80000 & all_times<90000);
    % times_non_art = all_spktimes(is_artifact & (which_chan==NSx(pos_chans_probe(2)).chan_ID));
    % times_non_art_plot=times_non_art(times_non_art>80000 & times_non_art<90000);
    % 
    % subplot(2,1,1)
    % xline(all_times_plot)
    % xlim([80000 90000])
    % subplot(2,1,2)
    % xline(times_non_art_plot,'color','r')
    % xlim([80000 90000])

    % Get number of parallel cores MATLAB can use
    b_parallel = true;
    num_cores = feature('numCores');
    % Check if there's enough spikes for parallel processing
    if num_cores * 100 < numel(all_spktimes) && b_parallel        
        % Split all_spktimes into num_cores parts
        split_size = floor(numel(all_spktimes) / num_cores);
        % split_idx = [1:split_size:numel(all_spktimes) numel(all_spktimes)+1];
        split_idx = [1:split_size:numel(all_spktimes)];
        split_idx(end) = numel(all_spktimes);
        num_cores = numel(split_idx) - 1;
        f_det(1:num_cores) = parallel.FevalFuture;
        
        for i = 1:num_cores
            start_idx = split_idx(i);
            end_idx = split_idx(i+1);
            
            % Adjust end_idx to include buffer spikes
            buffer_spks = 0;
            if end_idx < numel(all_spktimes)
                while end_idx + buffer_spks < numel(all_spktimes) && ...
                      all_spktimes(end_idx + buffer_spks + 1) <= all_spktimes(end_idx) + t_win
                    buffer_spks = buffer_spks + 1;
                end
            end
            
            spktimes = all_spktimes(start_idx:end_idx + buffer_spks);
            whichchan = which_chan(start_idx:end_idx + buffer_spks);
            f_det(i) = parfeval(@detect_artifacts, 1, start_idx, split_size, ...
                                spktimes, whichchan, t_win, bundle_min_art);
        end
        
        for i = 1:num_cores
            [completedIdx, art_idxs] = fetchNext(f_det);
            artifact_idxs = [artifact_idxs art_idxs];
        end
    else
        for ispk = 1:numel(all_spktimes)
            which_spks = find(all_spktimes >= all_spktimes(ispk) & all_spktimes < all_spktimes(ispk) + t_win);
            if numel(unique(which_chan(which_spks))) >= bundle_min_art
                artifact_idxs = [artifact_idxs which_spks];
            end
        end
    end

    artifact_idxs = unique(artifact_idxs);
    is_artifact(artifact_idxs) = true;

    for k= 1:length(pos_chans_probe)
        ch_lbl = NSx(pos_chans_probe(k)).output_name;
        ch_id  = NSx(pos_chans_probe(k)).chan_ID;
        SPK = load(sprintf('%s_spikes.mat', ch_lbl));
        if isfield(SPK,'index_all')
            spikes_all = SPK.spikes_all;
            index_all = SPK.index_all;
        else
            spikes_all = SPK.spikes;
            index_all = SPK.index;
        end
        par = SPK.par;
        par.t_win = t_win;
        par.bundle_min_art = bundle_min_art;

        index = all_spktimes(~is_artifact & (which_chan==ch_id));
        mask_nonart = ismember(index_all,index);
        spikes_coll_only = spikes_all(mask_nonart,:);

        % make_plots(spikes, spikes_all, mask_nonart, ch_lbl, b_parallel)
        
        save(sprintf('%s_spikes.mat', ch_lbl), ...
             "index", "spikes_coll_only", "index_all", "spikes_all", "par", "mask_nonart", "-append") 
        fprintf("%d/%d artifact spikes in %s\n", ...
            numel(index_all)-numel(index), numel(index_all), ch_lbl);
    end
    bundles_collisions_toc = toc(bundles_collisions_tic);
    fprintf("Collision detection (%s) took %s seconds. Total artifacts:%d/%d(%2.2f%%) \n", ...
        NSx(posch).bundle, num2str(bundles_collisions_toc, '%2.2f'), ...
        numel(artifact_idxs), numel(all_spktimes), numel(artifact_idxs)/numel(all_spktimes)*100);
end
separate_collisions_toc = toc(separate_collisions_tic);
    fprintf("separate_collisions DONE in %s seconds. \n", ...
        num2str(separate_collisions_toc, '%2.2f'));

end

function artifact_idxs = detect_artifacts(split_idx, splitsize, spktimes, ...
                                            whichchan,t_win,bundle_min_art)
    artifact_idxs = [];
    for ispk=1:splitsize
        which_spks = find(spktimes>=spktimes(ispk) & spktimes<spktimes(ispk)+t_win);
        if numel(unique(whichchan(which_spks)))>=bundle_min_art
            which_spks = which_spks + split_idx - 1;
            artifact_idxs = [artifact_idxs which_spks];
        end
    end
end

function make_plots(spikes, spikes_all, mask_nonart, ch_lbl, b_parallel)
    % plot the spikes
    f = figure('visible','off');
    for i = 1:size(spikes,1)
        plot(spikes(i,:))
        hold on
    end
    if b_parallel
        ext_lbl = 'parallel';
    else
        ext_lbl = 'nonparallel';
    end
    % save the figure as png
    saveas(f, sprintf('%s_spikes_%s.jpg', ch_lbl, ext_lbl))
    

    % plot the artifacts
    f = figure('visible','off');
    for i = 1:size(spikes_all,1)
        if mask_nonart(i)
            continue
        end
        plot(spikes_all(i,:))
        hold on
    end
    % save the figure
    saveas(f, sprintf('%s_artifacts_%s.jpg',NSx(pos_chans_probe(k)).output_name, ext_lbl))
end

