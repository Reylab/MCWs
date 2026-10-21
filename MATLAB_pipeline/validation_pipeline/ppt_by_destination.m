function ppt_by_destination(t, Wv, oc, oc_match, onm, tm, d, state, two, side, cc, ttl)
% PPT_BY_DESTINATION  Split a venn region by where its spikes are in the other sort -- no spike within tol /
% that sort's cluster 0 / each of its other classes -- into folder D: waveforms_by_destination.png (all, then
% each group, with the templates TM), one raster per group (STATE / SIDE = the region's full-run windows; TWO =
% also the default-window version) and counts.csv.
%   t, Wv     spike times (ms) and waveforms of the region
%   oc        the other sort's class of each spike (NaN = no spike there); oc_match = its matched class
%   onm       the other sort's name (labels)
if ~exist(d, 'dir'), mkdir(d); end
grp = struct('name', 'all', 'W', Wv, 'color', cc); cnt = {'all', numel(t)};
tag = lower(regexprep(onm, '\s.*', ''));        % 'python' / 'gt' / 'wave_clus' for the folder names
sets = {sprintf('not detected by %s', onm), isnan(oc), ['not_in_' tag]; sprintf('%s cluster 0', onm), oc == 0, ['in_' tag '_cluster0']};
for k = setdiff(unique(oc(~isnan(oc)))', [0 oc_match])
    sets(end + 1, :) = {sprintf('%s class %d', onm, k), oc == k, sprintf('in_%s_c%d', tag, k)}; %#ok<AGROW>
end
for i = 1:size(sets, 1)
    m = sets{i, 2}; cnt(end + 1, :) = {sets{i, 1}, nnz(m)}; %#ok<AGROW>
    if ~any(m), continue; end
    grp(end + 1) = struct('name', sets{i, 1}, 'W', Wv(m, :), 'color', cc * 0.7); %#ok<AGROW>
    nm = sprintf('%02d_%s', i, sets{i, 3});
    plot_subset_rasters([], 'dir', state, 'times', t(m), 'name', nm, 'side', side, 'out_dir', d);
    if two
        plot_subset_rasters([], 'dir', state, 'times', t(m), 'name', [nm '_default_window'], 'side', side, 'out_dir', d, 'window', 'default');
    end
end
plot_subset_waveforms(grp, tm, ttl, fullfile(d, 'waveforms_by_destination.png'));
writetable(cell2table(cnt, 'VariableNames', {'where', 'n_spikes'}), fullfile(d, 'counts.csv'));
end

