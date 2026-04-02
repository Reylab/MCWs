function [detecctions_trial_new,is_art] = remove_collisions_bundle(chan_label,detecctions_trial)

t_win = 0.5*30; % 0.5ms in samples
bundle_min_art = 6/8;
num_chans = length(chan_label);
chan_bundle = cell(num_chans,1);

for ii=1:num_chans
    tt = regexp(chan_label{ii},'\d+','once');
    if ~isempty(tt)
        chan_bundle{ii} = chan_label{ii}(1:tt-1);
    else
        chan_bundle{ii} = chan_label{ii};
    end
end

bundles_to_explore = unique(chan_bundle);
detecctions_trial_new = cell(size(detecctions_trial));
all_spktimes = cell(1,length(bundles_to_explore));
which_chan = cell(1,length(bundles_to_explore));
which_bin = cell(1,length(bundles_to_explore));
artifact_bin = cell(1,length(bundles_to_explore));
is_art = cell(size(detecctions_trial));

for ibun = 1:length(bundles_to_explore)    
    pos_chans_probe = find(strcmp(chan_bundle,bundles_to_explore{ibun}));

    for k= 1:length(pos_chans_probe)
        all_spktimes{ibun} = [all_spktimes{ibun} detecctions_trial{pos_chans_probe(k)}];
        which_chan{ibun}  = [which_chan{ibun}  pos_chans_probe(k)*ones(size(detecctions_trial{pos_chans_probe(k)}))];
    end
    [all_spktimes{ibun} ,II] = sort(all_spktimes{ibun} );
    if numel(all_spktimes{ibun}) > 8
        which_chan{ibun}  = which_chan{ibun}(II);
        [spike_timeline,~,which_bin{ibun}] = histcounts(all_spktimes{ibun}, min(all_spktimes{ibun}):t_win:max(all_spktimes{ibun}));
        artifact_bin{ibun} = find(spike_timeline>=bundle_min_art*length(pos_chans_probe));
    else
        artifact_bin{ibun} = [];
        fprintf('Not enough spikes(numspks:%d) in %s to detect collisions.\n',numel(all_spktimes{ibun}), bundles_to_explore{ibun});
    end
end
for ibun = 1:length(bundles_to_explore)
    pos_chans_probe = find(strcmp(chan_bundle,bundles_to_explore{ibun}));

    for k= 1:length(pos_chans_probe)
        if numel(all_spktimes{ibun}) > 8
            detecctions_trial_new{pos_chans_probe(k)}  = all_spktimes{ibun}(~ismember(which_bin{ibun},artifact_bin{ibun}) & (which_chan{ibun}==pos_chans_probe(k)));
            is_art{pos_chans_probe(k)} = ~ismember(detecctions_trial{pos_chans_probe(k)},detecctions_trial_new{pos_chans_probe(k)});
        end
    end
end