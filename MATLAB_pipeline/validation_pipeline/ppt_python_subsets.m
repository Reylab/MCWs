function ppt_python_subsets(session, multi, run_label, which)
% PPT_PYTHON_SUBSETS  The python side of a for ppt report (build_ppt_report must have run first):
%   <run folder>/03b_python_only/   python class spikes the wave_clus class lacks and python did not
%                                   quarantine (the venn's MCWs-only region)
%   <run folder>/03c_rescued_only/  python class spikes that were quarantined and rescued into the class,
%                                   and that the wave_clus class lacks (the venn's quarantine x MCWs region)
%   <run folder>/03d_quarantine_only/  python quarantined spikes in neither the python class nor the wave_clus
%                                   class (the venn's quarantine-only region), then by quarantine property
%                                   (waveforms + rasters per property, counts.csv, counts_by_combination.csv)
% WHICH (optional): subset of {'03b', '03c', '03d'} to make (default all).
% Each: waveforms (all, then split by where they are in wave_clus) with the GT / wave_clus / python class
% templates, rasters (the full run's windows, plus the default-window version when the run has
% late-latency / long-duration stimuli) and a csv of the spikes.
%   ppt_python_subsets('pt4_ch259')                    % one GT unit in the channel
%   ppt_python_subsets('pt11emu17_ch321_class2', true)  % channel with several GT units (for ppt/GT c<g>)
if nargin < 2, multi = false; end
if nargin < 3 || isempty(run_label), run_label = 'notch'; end
if nargin < 4, which = {'03b', '03c', '03d'}; end
here = fileparts(mfilename('fullpath')); addpath(here, fullfile(fileparts(here), 'real_response_analysis'));
cfg = validation_sessions(session); g = cfg.gt.class; ch = cfg.ch; root = cfg.root;
pp = ppt_paths(cfg, multi); out = pp.rep;
M = load(fullfile(out, sprintf('membership_ch%d.mat', ch)));
nm = matlab.lang.makeValidName(['wc_' run_label]); R = ['wave_clus ' run_label];
py_c = M.classes.python; wc_c = M.classes.(nm);
py = pp.py;
% python subsets use python's own full-run windows: the arm table of the wave_clus vs python run (08)
state = fullfile(pp.wcval, sprintf('validation_wave_clus_vs_python_c%d', py_c));
w = load(fullfile(state, 'validation_state.mat'), 'ref_tables_default');
two = isfield(w, 'ref_tables_default') && ~isempty(w.ref_tables_default);
col.wc = [0.00 0.45 0.70]; col.py = [0.90 0.60 0.00]; col.q = [0.80 0.15 0.15];
lab = sprintf('wave_clus %s c%d', run_label, wc_c);
tm = struct('name', {sprintf('GT c%d', g), lab, sprintf('python c%d', py_c)}, ...
    'wave', {M.templates.gt(g, :), M.templates.(nm)(wc_c, :), M.templates.python(py_c, :)}, ...
    'color', {[0 0 0], col.wc, col.py}, 'style', {'-', '--', ':'});

P = M.python; c = P.([nm '_class']); wave = M.waves.python;
not_wc = isnan(c) | c ~= wc_c;
sets = {'03b_python_only', sprintf('python c%d only (not quarantined)', py_c), P.class == py_c & ~P.quarantined & not_wc, col.py; ...
        '03c_rescued_only', sprintf('python c%d rescued only', py_c), P.class == py_c & P.quarantined & not_wc, col.q};
sets = sets(ismember(cellfun(@(x) x(1:3), sets(:, 1), 'UniformOutput', false), which), :);
for i = 1:size(sets, 1)
    [sub, ttl, m, cc] = sets{i, :};
    d = fullfile(out, R, sub); if ~exist(d, 'dir'), mkdir(d); end
    where = {sprintf('no wave_clus spike within %g ms', M.tol), isnan(c); 'wave_clus cluster 0', c == 0};
    for k = setdiff(unique(c(m & ~isnan(c)))', [0 wc_c])
        where(end + 1, :) = {sprintf('wave_clus class %d', k), c == k}; %#ok<AGROW>
    end
    grp = struct('name', ttl, 'W', wave(m, :), 'color', cc);
    for j = 1:size(where, 1)
        grp(end + 1) = struct('name', ['  ... ' where{j, 1}], 'W', wave(m & where{j, 2}, :), 'color', cc * 0.6); %#ok<AGROW>
    end
    plot_subset_waveforms(grp, tm, sprintf('ch%d: %s that %s lacks, and where they are in wave_clus', ch, ttl, lab), ...
        fullfile(d, ['waveforms_' sub(5:end) '.png']));
    writetable(P(m, :), fullfile(d, [sub(5:end) '_spikes.csv']));
    t = P.t(m); nmr = sprintf('python_c%d_%s', py_c, sub(5:end));
    if isempty(t), fprintf('[ppt_python_subsets] %s: no spikes, no rasters\n', nmr); continue; end
    plot_subset_rasters([], 'dir', state, 'times', t, 'name', nmr, 'side', 'arm', 'out_dir', d);
    if two
        plot_subset_rasters([], 'dir', state, 'times', t, 'name', [nmr '_default_window'], 'side', 'arm', ...
            'out_dir', d, 'window', 'default');
    end
end
if ismember('03d', which)
    quarantine_only(P, c, wc_c, py_c, wave, tm, ch, lab, state, two, fullfile(out, R, '03d_quarantine_only'), col);
end
fprintf('[ppt_python_subsets] %s done -> %s\n', session, fullfile(out, R));
end

function quarantine_only(P, c, wc_c, py_c, wave, tm, ch, lab, state, two, d, col)
% python quarantined spikes in neither python's class nor the wave_clus class (the venn's quarantine-only
% region): all of them, then by quarantine property (a spike can fail several)
if ~exist(d, 'dir'), mkdir(d); end
m = P.quarantined & P.class ~= py_c & (isnan(c) | c ~= wc_c); rows = find(m);
raster = @(t, name) sub_rasters(state, t, name, d, two);
pq = P.Properties.VariableNames(startsWith(P.Properties.VariableNames, 'q_'));
grp = struct('name', sprintf('python quarantine only (not python c%d, not %s)', py_c, lab), 'W', wave(rows, :), 'color', col.q);
cnt = {'all (python quarantine only)', numel(rows)};
raster(P.t(rows), '00_python_quarantine_only');
for j = 1:numel(pq)
    hit = P.(pq{j})(rows); pn = strrep(pq{j}(3:end), '_', ' ');
    cnt(end + 1, :) = {pn, nnz(hit)}; %#ok<AGROW>
    if ~any(hit), continue; end
    grp(end + 1) = struct('name', ['quarantined for ' pn], 'W', wave(rows(hit), :), 'color', col.q * 0.75); %#ok<AGROW>
    raster(P.t(rows(hit)), sprintf('%02d_%s', j, pq{j}(3:end)));
end
resc = P.rescued(rows);
cnt(end + 1:end + 2, :) = {'rescued into another python class', nnz(resc); 'not rescued (python cluster 0)', nnz(~resc)};
if any(resc)
    grp(end + 1) = struct('name', 'rescued into another python class', 'W', wave(rows(resc), :), 'color', [0.35 0.35 0.35]);
    grp(end + 1) = struct('name', 'not rescued', 'W', wave(rows(~resc), :), 'color', col.q * 0.6);
    raster(P.t(rows(resc)), '90_rescued_other_class'); raster(P.t(rows(~resc)), '91_not_rescued');
end
plot_subset_waveforms(grp, tm, sprintf(['ch%d: python quarantined spikes in neither python c%d nor %s, ' ...
    'then by quarantine property (a spike can fail several)'], ch, py_c, lab), fullfile(d, 'waveforms_by_quarantine_property.png'));
F = false(numel(rows), numel(pq)); for j = 1:numel(pq), F(:, j) = P.(pq{j})(rows); end
[u, ~, k] = unique(F, 'rows');
combo = arrayfun(@(x) strjoin(strrep(erase(pq(u(x, :)), 'q_'), '_', ' '), ' + '), (1:size(u, 1))', 'UniformOutput', false);
combo(cellfun(@isempty, combo)) = {'(no property flag)'};
writetable(cell2table(cnt, 'VariableNames', {'group', 'n_spikes'}), fullfile(d, 'counts.csv'));
if ~isempty(rows)
    writetable(table(combo, accumarray(k, 1), 'VariableNames', {'properties', 'n_spikes'}), fullfile(d, 'counts_by_combination.csv'));
end
writetable(P(rows, :), fullfile(d, 'quarantine_only_spikes.csv'));
end

function sub_rasters(state, t, name, d, two)
if isempty(t), fprintf('[ppt_python_subsets] %s: no spikes, no rasters\n', name); return; end
plot_subset_rasters([], 'dir', state, 'times', t, 'name', name, 'side', 'arm', 'out_dir', d);
if two
    plot_subset_rasters([], 'dir', state, 'times', t, 'name', [name '_default_window'], 'side', 'arm', 'out_dir', d, 'window', 'default');
end
end
