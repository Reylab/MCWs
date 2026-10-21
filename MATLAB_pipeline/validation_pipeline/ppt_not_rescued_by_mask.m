function ppt_not_rescued_by_mask(session, multi, pages_only)
% PPT_NOT_RESCUED_BY_MASK  The quarantined spikes that were NEVER rescued into any cluster (python cluster 0),
% split by the mask (quarantine property) that quarantined them -> <region>/not_rescued_by_mask/:
%   wave_clus notch/05_wave_clus_x_quarantine   wave_clus class spikes python quarantined + left in cluster 0
%   wave_clus notch/03d_quarantine_only          python quarantined spikes in neither class, left in cluster 0
%   09_gt_vs_python/04_gt_x_quarantine           GT class spikes python quarantined + left in cluster 0
%   09_gt_vs_python/06_quarantine_only           python quarantined spikes in neither class, left in cluster 0
% Mask flags are python's (on the python spike itself, or on the python spike paired with a wave_clus / GT
% spike). Each folder: waveforms_not_rescued_by_mask.png (all, then each mask, templates on top), one raster
% per mask with the windows of the sort the spikes come from (wave_clus / python / GT matched class) + the
% default-window version, counts.csv, counts_by_combination.csv (a spike can fail several masks).
% Also in each folder (ppt_quarantine_pages): distance_to_templates_by_mask.png (distance to the nearest class
% template in rescue radii) and quarantine_values_by_mask.png (the value that quarantined each spike vs its threshold).
%   ppt_not_rescued_by_mask('pt5emu4_ch271_class2', false)
%   ppt_not_rescued_by_mask('pt5emu4_ch271_class2', false, true)   % only the two quarantine pages
if nargin < 2, multi = false; end
if nargin < 3, pages_only = false; end
here = fileparts(mfilename('fullpath')); addpath(here, fullfile(fileparts(here), 'real_response_analysis'));
cfg = validation_sessions(session); g = cfg.gt.class; root = cfg.root; ch = cfg.ch;
pp = ppt_paths(cfg, multi); rep = pp.rep;
M = load(fullfile(rep, sprintf('membership_ch%d.mat', ch)));
G = M.gt; P = M.python; W = M.wc_notch; py_c = M.classes.python; wc_c = M.classes.wc_notch;
py = pp.py;
Tq = load(fullfile(root, py, cfg.gt.times_file), 'quarantine_properties', 'par', 'spikes', 'cluster_class', 'forced', 'quarantine');
S_wc = fullfile(pp.wcval, sprintf('validation_wave_clus_vs_python_c%d', py_c));   % ref wave_clus, arm python
S_gt = fullfile(pp.val, sprintf('validation_gt_vs_%s_class%d', pp.pyname, g));                            % ref GT, arm python
B = readtable(fullfile(pp.val, sprintf('venn_gt_vs_python_class%d', g), 'arm_classes.csv')); pg = B.class(1);
tm = struct('name', {sprintf('GT c%d', g), sprintf('wave_clus notch c%d', wc_c), sprintf('python c%d', py_c)}, ...
    'wave', {M.templates.gt(g, :), M.templates.wc_notch(wc_c, :), M.templates.python(py_c, :)}, ...
    'color', {[0 0 0], [0 .45 .7], [.9 .6 0]}, 'style', {'-', '--', ':'});

% python spike row of every spike in each region, and "never rescued" = python cluster 0 after everything
in_wc = W.class == wc_c; in_gt = G.class == g;
not_wc = isnan(P.wc_notch_class) | P.wc_notch_class ~= wc_c; not_gt = isnan(P.gt_class) | P.gt_class ~= g;
r05 = find(in_wc & W.python_quarantined & W.python_class == 0);                % wave_clus spikes, python left them in 0
r3d = find(P.quarantined & P.class == 0 & not_wc);                            % python spikes in no class
r04 = find(in_gt & G.python_quarantined & G.python_class == 0);               % GT spikes, python left them in 0
r06 = find(P.quarantined & P.class == 0 & not_gt);
R = {fullfile(rep, 'wave_clus notch', '05_wave_clus_x_quarantine'), W.t(r05), M.waves.wc_notch(r05, :), W.python_idx(r05), S_wc, 'gt', 'wave_clus'; ...
     fullfile(rep, 'wave_clus notch', '03d_quarantine_only'),       P.t(r3d), M.waves.python(r3d, :),   r3d,               S_wc, 'arm', 'python'; ...
     fullfile(rep, '09_gt_vs_python', '04_gt_x_quarantine'),        G.t(r04), M.waves.gt(r04, :),       G.python_idx(r04), S_gt, 'gt', 'GT'; ...
     fullfile(rep, '09_gt_vs_python', '06_quarantine_only'),        P.t(r06), M.waves.python(r06, :),   r06,               S_gt, 'arm', 'python'};
pq = P.Properties.VariableNames(startsWith(P.Properties.VariableNames, 'q_'));
for i = 1:size(R, 1)
    [base, t, Wv, pidx, st, side, who] = R{i, :};
    out = fullfile(base, 'not_rescued_by_mask'); if ~exist(out, 'dir'), mkdir(out); end
    [~, rn0] = fileparts(base);
    ppt_quarantine_pages(pidx, P, Tq, out, sprintf('ch%d %s: %s spikes python quarantined and never rescued', ch, strrep(rn0(4:end), '_', ' '), who));
    if pages_only, continue; end
    w = load(fullfile(st, 'validation_state.mat'), 'ref_tables_default'); two = isfield(w, 'ref_tables_default') && ~isempty(w.ref_tables_default);
    [~, rn] = fileparts(base);
    grp = struct('name', sprintf('%s: never rescued (python cluster 0), all', strrep(rn(4:end), '_', ' ')), 'W', Wv, 'color', [.5 .5 .5]);
    cnt = {'all never rescued', numel(t)}; F = false(numel(t), numel(pq));
    sub(st, t, '00_all_not_rescued', out, two, side);
    for j = 1:numel(pq)
        hit = P.(pq{j})(pidx); F(:, j) = hit; pn = strrep(pq{j}(3:end), '_', ' ');
        cnt(end + 1, :) = {pn, nnz(hit)}; %#ok<AGROW>
        if ~any(hit), continue; end
        grp(end + 1) = struct('name', ['quarantined for ' pn], 'W', Wv(hit, :), 'color', [.8 .15 .15]); %#ok<AGROW>
        sub(st, t(hit), sprintf('%02d_%s', j, pq{j}(3:end)), out, two, side);
    end
    plot_subset_waveforms(grp, tm, sprintf(['ch%d: %s spikes in %s that python quarantined and never rescued, by mask ' ...
        '(a spike can fail several)'], ch, who, strrep(rn(4:end), '_', ' ')), fullfile(out, 'waveforms_not_rescued_by_mask.png'));
    writetable(cell2table(cnt, 'VariableNames', {'mask', 'n_spikes'}), fullfile(out, 'counts.csv'));
    if ~isempty(t)
        [u, ~, k] = unique(F, 'rows');
        combo = arrayfun(@(x) strjoin(strrep(erase(pq(u(x, :)), 'q_'), '_', ' '), ' + '), (1:size(u, 1))', 'UniformOutput', false);
        combo(cellfun(@isempty, combo)) = {'(no mask flag)'};
        writetable(table(combo, accumarray(k, 1), 'VariableNames', {'masks', 'n_spikes'}), fullfile(out, 'counts_by_combination.csv'));
    end
    fprintf('[ppt_not_rescued_by_mask] %s: %d never rescued -> %s\n', rn, numel(t), out);
end
end

function sub(st, t, name, out, two, side)
if isempty(t), fprintf('[ppt_not_rescued_by_mask] %s: no spikes, no rasters\n', name); return; end
plot_subset_rasters([], 'dir', st, 'times', t, 'name', name, 'side', side, 'out_dir', out);
if two, plot_subset_rasters([], 'dir', st, 'times', t, 'name', [name '_default_window'], 'side', side, 'out_dir', out, 'window', 'default'); end
end
