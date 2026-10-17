function R = export_validation_results(d)
% EXPORT_VALIDATION_RESULTS  Every number behind the raster pages of a run,
% GT, each arm and each subset, in one long format, so runs (trials,
% channels, settings) can be stacked and compared directly.
%
%   export_validation_results            % from inside a validation out_dir
%   export_validation_results(out_dir)
%   (run_validation calls it last, stage 'results')
%
% unit       - 'gt', '<arm>', or '<arm>/<subset>' (e.g. python_sd2/arm_only)
% unit_kind  - 'gt' | 'arm' | 'subset'
% ref_unit   - whose latency / windows the row is scored on (itself for gt
%              and arms; gt or the arm for a subset)
%
% OUTPUTS (under out_dir/results)
%   per_stimulus.csv  one row per (unit, stimulus): ntrials, n_response
%                     (spikes in the green window, all trials), mean_post,
%                     median_post, zscore, p_value_sign, sign_rank,
%                     mu_base, std_base, onset (lat), dura, tons, good_lat,
%                     ifr_thr, win_lo/win_hi (green lines), post_lo/post_hi
%                     (median/sign-test window), base_lo/base_hi, n_spikes
%                     (whole unit), plus run/channel/class identifiers
%   per_unit.csv      one row per unit: n_spikes, n in any window, mu/sd base,
%                     n stimuli with sign p <= 0.05 / zscore >= 5, medians
%   per_trial.csv     one row per (unit, stimulus, trial): spike counts in the
%                     baseline, post and green windows (what median_post,
%                     the sign test and n_response/mean_post are made of)
%   rasters.mat       table `spikes`: (unit, stim_number, trial, t_ms) for
%                     every raster tick (ms from onset), plus the tables above
%
% Rows are only as good as the data_table.mat each page saved: rerun the
% raster stages first if those are older than this export.

if nargin < 1 || isempty(d), d = pwd; end
here = fileparts(mfilename('fullpath'));
addpath(fullfile(fileparts(here), 'real_response_analysis'));

st = load_validation_state(d);
cfg = st.cfg;
assert(~isempty(st.ref_tables), 'export_validation_results: no raster tables in %s -- run the rasters first.', st.state_dir);
ch = cfg.ch; cf = sprintf('chan%d', ch);
[~, run_name] = fileparts(cfg.out_dir);

% units: GT, arms (their rasters are the class field of their own grapes),
% subsets (the 'mu' field of their synthetic grapes)
U = struct('unit', {}, 'kind', {}, 'ref_unit', {}, 'class', {}, 'T', {}, 'raster', {});
gt_grapes = load(fullfile(cfg.gt.dir, cfg.gt.grapes_file), 'rasters');
U(end+1) = unit_row(cfg.gt.name, 'gt', cfg.gt.name, cfg.gt.class, st.ref_tables{1}, ...
    gt_grapes.rasters.(cf).(sprintf('class%d', cfg.gt.class)));
for k = 1:numel(cfg.arms)
    nm = cfg.arms(k).name;
    g = load(fullfile(cfg.arms(k).dir, cfg.arms(k).grapes_file), 'rasters');
    U(end+1) = unit_row(nm, 'arm', nm, st.arm_classes(k), st.ref_tables{1 + k}, ...
        g.rasters.(cf).(sprintf('class%d', st.arm_classes(k)))); %#ok<AGROW>
    sdir = fullfile(cfg.out_dir, 'subset_rasters', nm);
    if ~exist(sdir, 'dir'), continue; end
    sf = fullfile(cfg.out_dir, 'subsets', sprintf('subsets_%s.mat', nm));
    S = struct('subsets', struct());
    if exist(sf, 'file'), S = load(sf, 'subsets'); end
    for e = dir(sdir)'
        dtf = fullfile(sdir, e.name, 'data_table.mat');
        if ~e.isdir || startsWith(e.name, '.') || ~exist(dtf, 'file'), continue; end
        dt = getfield(load(dtf, 'data_table'), 'data_table'); %#ok<GFLD>
        sg = load(fullfile(sdir, e.name, 'grapes_blanks.mat'), 'rasters');
        ref = nm;
        if isfield(S.subsets, e.name) && strcmp(S.subsets.(e.name).side, 'gt'), ref = cfg.gt.name; end
        U(end+1) = unit_row(sprintf('%s/%s', nm, e.name), 'subset', ref, NaN, dt, sg.rasters.(cf).mu); %#ok<AGROW>
    end
end

per_stim = table(); per_trial = table(); spikes = table(); per_unit = table();
for u = 1:numel(U)
    T = U(u).T;
    n = height(T);
    cols = {'stim_number', 'stim_name', 'ntrials', 'n_response', 'mean_post', 'median_post', 'zscore', ...
            'p_value_sign', 'mu_base', 'std_base', 'onset', 'dura', 'tons', 'good_lat', 'ifr_thr', ...
            'win_lo', 'win_hi', 'post_lo', 'post_hi', 'base_lo', 'base_hi', 'cluster_total'};
    P = table();
    P.run = repmat({run_name}, n, 1);
    P.channel = repmat(ch, n, 1);
    P.unit = repmat({U(u).unit}, n, 1);
    P.unit_kind = repmat({U(u).kind}, n, 1);
    P.ref_unit = repmat({U(u).ref_unit}, n, 1);
    P.class = repmat(U(u).class, n, 1);
    for c = cols
        if ismember(c{1}, T.Properties.VariableNames)
            v = T.(c{1});
        elseif strcmp(c{1}, 'stim_name')
            v = repmat({''}, n, 1);
        else
            v = nan(n, 1);
        end
        P.(c{1}) = v;
    end
    P.sign_rank = arrayfun(@(i) rank_of(P.p_value_sign, i), (1:n)');
    P = renamevars(P, 'cluster_total', 'n_spikes');
    if iscategorical(P.stim_name) || isstring(P.stim_name), P.stim_name = cellstr(P.stim_name); end

    % per trial: counts in the baseline / post / green windows, and the ticks
    tr_u = {}; sp_u = {};
    for i = 1:n
        trials = U(u).raster.stim{P.stim_number(i)};
        if ~iscell(trials), trials = num2cell(trials, 2); end
        trials = cellfun(@(x) reshape(x(isfinite(x)), [], 1), trials(:), 'UniformOutput', false);
        nt = numel(trials);
        cnt = @(lo, hi) cellfun(@(x) nnz(x > lo & x < hi), trials);
        tr_u{end+1} = table(repmat({U(u).unit}, nt, 1), repmat(P.stim_number(i), nt, 1), (1:nt)', ...
            cnt(P.base_lo(i), P.base_hi(i)), cnt(P.post_lo(i), P.post_hi(i)), ...
            cellfun(@(x) nnz(x >= P.win_lo(i) & x <= P.win_hi(i)), trials), ...
            'VariableNames', {'unit', 'stim_number', 'trial', 'n_base', 'n_post', 'n_window'}); %#ok<AGROW>
        ns = cellfun(@numel, trials);
        sp_u{end+1} = table(repmat({U(u).unit}, sum(ns), 1), repmat(P.stim_number(i), sum(ns), 1), ...
            repelem((1:nt)', ns), vertcat(trials{:}), ...
            'VariableNames', {'unit', 'stim_number', 'trial', 't_ms'}); %#ok<AGROW>
    end
    per_stim  = [per_stim; P]; %#ok<AGROW>
    per_trial = [per_trial; vertcat(tr_u{:})]; %#ok<AGROW>
    spikes    = [spikes; vertcat(sp_u{:})]; %#ok<AGROW>

    per_unit = [per_unit; table({run_name}, ch, {U(u).unit}, {U(u).kind}, {U(u).ref_unit}, U(u).class, ...
        P.n_spikes(1), sum(P.n_response, 'omitnan'), P.mu_base(1), P.std_base(1), n, ...
        nnz(P.p_value_sign <= 0.05), nnz(P.zscore >= 5), median(P.median_post, 'omitnan'), ...
        median(P.zscore, 'omitnan'), median(P.mean_post, 'omitnan'), ...
        'VariableNames', {'run', 'channel', 'unit', 'unit_kind', 'ref_unit', 'class', 'n_spikes', ...
        'n_response_total', 'mu_base', 'std_base', 'n_stims', 'n_sign_p05', 'n_z_ge5', ...
        'median_median_post', 'median_zscore', 'median_mean_post'})]; %#ok<AGROW>
end

out_dir = fullfile(cfg.out_dir, 'results');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
writetable(per_stim,  fullfile(out_dir, 'per_stimulus.csv'));
writetable(per_unit,  fullfile(out_dir, 'per_unit.csv'));
writetable(per_trial, fullfile(out_dir, 'per_trial.csv'));
save(fullfile(out_dir, 'rasters.mat'), 'spikes', 'per_stim', 'per_unit', 'per_trial', 'cfg', '-v7.3');
fprintf('[results] %d units, %d stimulus rows, %d trial rows, %d raster ticks -> %s\n', ...
    numel(U), height(per_stim), height(per_trial), height(spikes), out_dir);
R = struct('per_stim', per_stim, 'per_unit', per_unit, 'per_trial', per_trial, 'spikes', spikes);
end

function u = unit_row(unit, kind, ref_unit, cls, T, raster)
    u = struct('unit', unit, 'kind', kind, 'ref_unit', ref_unit, 'class', cls, 'T', T, 'raster', raster);
end

function r = rank_of(p, i)
% 1 = smallest p among the unit's stimuli, ties share a rank (as on the pages)
    if ~isfinite(p(i)), r = NaN; return; end
    r = nnz(p(isfinite(p)) < p(i)) + 1;
end
