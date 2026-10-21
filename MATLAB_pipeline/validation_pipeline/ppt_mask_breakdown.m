function ppt_mask_breakdown(t, Wv, pidx, P, tm, out, state, two, side, ttl)
% PPT_MASK_BREAKDOWN  One set of quarantined spikes split by the python mask(s) that quarantined them, into
% folder OUT: waveforms_by_mask.png (all, then each mask, templates TM on top), a raster for all and one per mask
% (STATE / SIDE = the windows of the sort the spikes come from; TWO = also the default-window version),
% counts.csv and counts_by_combination.csv (a spike can fail several masks).
%   t, Wv   spike times (ms) and waveforms of the set
%   pidx    row in the python table P of each spike's python spike (itself, or the one paired with it)
if ~exist(out, 'dir'), mkdir(out); end
pq = P.Properties.VariableNames(startsWith(P.Properties.VariableNames, 'q_'));
grp = struct('name', 'all', 'W', Wv, 'color', [.5 .5 .5]); cnt = {'all', numel(t)}; F = false(numel(t), numel(pq));
sub(state, t, '00_all', out, two, side);
for j = 1:numel(pq)
    hit = P.(pq{j})(pidx); F(:, j) = hit; pn = strrep(pq{j}(3:end), '_', ' ');
    cnt(end + 1, :) = {pn, nnz(hit)}; %#ok<AGROW>
    if ~any(hit), continue; end
    grp(end + 1) = struct('name', ['quarantined for ' pn], 'W', Wv(hit, :), 'color', [.8 .15 .15]); %#ok<AGROW>
    sub(state, t(hit), sprintf('%02d_%s', j, pq{j}(3:end)), out, two, side);
end
plot_subset_waveforms(grp, tm, [ttl ' (a spike can fail several masks)'], fullfile(out, 'waveforms_by_mask.png'));
writetable(cell2table(cnt, 'VariableNames', {'mask', 'n_spikes'}), fullfile(out, 'counts.csv'));
if ~isempty(t)
    [u, ~, k] = unique(F, 'rows');
    combo = arrayfun(@(x) strjoin(strrep(erase(pq(u(x, :)), 'q_'), '_', ' '), ' + '), (1:size(u, 1))', 'UniformOutput', false);
    combo(cellfun(@isempty, combo)) = {'(no mask flag)'};
    writetable(table(combo, accumarray(k, 1), 'VariableNames', {'masks', 'n_spikes'}), fullfile(out, 'counts_by_combination.csv'));
end
end

function sub(st, t, name, out, two, side)
if isempty(t), return; end
plot_subset_rasters([], 'dir', st, 'times', t, 'name', name, 'side', side, 'out_dir', out);
if two, plot_subset_rasters([], 'dir', st, 'times', t, 'name', [name '_default_window'], 'side', side, 'out_dir', out, 'window', 'default'); end
end
