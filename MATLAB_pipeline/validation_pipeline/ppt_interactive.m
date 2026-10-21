%% PPT_INTERACTIVE  Step-through version of the for ppt report, for one GT unit.
% Run one section at a time (Ctrl+Enter in the editor, or "Run Section"), change the subsets / settings and
% re-run only the sections you need. Nothing here deletes anything; figures go to OUT below.
% Full instructions: README_for_ppt.md (same folder).
%
% Section order:
%   0  setup (channel / GT unit / output folder)          5  waveforms of any subsets
%   1  everything automatic for this unit (optional)      6  rasters of any subsets (right windows per sort)
%   2  load the membership (every spike, every sort)       7  rasters of a whole class (its own windows)
%   3  the three comparison runs (= the raster windows)    8  bookkeeping check vs venns / raster pages
%   4  define subsets (examples - edit these)

%% 0  setup -------------------------------------------------------------------------------------------------
V = fullfile(mcws_paths().repo, 'validation_pipeline');   % needs validation_pipeline on the path (for mcws_paths)
addpath(V, fullfile(fileparts(V), 'real_response_analysis'));
session = 'pt4_ch259';          % any id from validation_sessions() -- validation_sessions() lists them
multi   = false;                % true when the channel has more than one GT unit (report goes to for ppt/GT c<g>)
cfg  = validation_sessions(session);
root = cfg.root; g = cfg.gt.class; ch = cfg.ch; tf = cfg.gt.times_file;
wcf  = fullfile(root, 'wave_clus notch');                    % the wave_clus run compared (GT notch)
d = dir(fullfile(root, 'python_recluster_*_second_pass')); py = d(end).name;   % newest python arm (GT notch)
rep  = fullfile(root, 'for ppt'); if multi, rep = fullfile(rep, sprintf('GT c%d', g)); end
OUT  = fullfile(rep, 'my_subsets');                          % where the sections below write
if ~exist(OUT, 'dir'), mkdir(OUT); end
fprintf('%s: %s, GT class %d, python %s\n', session, root, g, py);

%% 1  (optional) the whole automatic report for this unit -----------------------------------------------------
% grapes for the notch run, its venns, build_ppt_report (01-08), python subsets (03b-03d), GT comparisons (09).
% ~30-40 min. Skip this section if the report exists already.
% ppt_session(session, multi);
% ppt_check_unit(session, multi);

%% 2  membership: every spike of GT / python / wave_clus notch and where it is in the other two ---------------
% Built by the report (build_sort_membership). Tables M.gt, M.python, M.wc_notch, one row per spike:
%   t, class, <other>_class (NaN = no spike within tol), <other>_idx (row in the other table), <other>_dt,
%   python only: quarantined, rescued, forced, mahal_reassigned, q_<property> (quarantine flags)
%   <other>_quarantined = the paired python spike was quarantined
% M.waves.<sort> = the spikes (rows as the table), M.templates.<sort>(k, :) = class k mean,
% M.classes = the matched classes (gt, python, wc_notch).
M = load(fullfile(rep, sprintf('membership_ch%d.mat', ch)));
G = M.gt; P = M.python; W = M.wc_notch;
py_c = M.classes.python; wc_c = M.classes.wc_notch;
fprintf('GT c%d <-> python c%d <-> wave_clus notch c%d\n', g, py_c, wc_c);
% (to rebuild it, e.g. with another tolerance:)
% sorts = struct('name', {'gt', 'python', 'wc_notch'}, 'file', {fullfile(root, 'hand selected', tf), ...
%     fullfile(root, py, tf), fullfile(wcf, tf)});
% M = build_sort_membership(sorts, 'tol', 0.5, 'out_file', fullfile(OUT, 'membership_mine.mat'));

%% 3  the three full comparison runs -- their raster tables ARE the response windows --------------------------
% Every subset raster takes the windows of the FULL run of the sort its spikes come from:
%   side 'gt'  = the run's reference unit,  side 'arm' = the run's arm.
S_wc_py = fullfile(wcf, sprintf('validation_wave_clus_vs_python_c%d', py_c));   % ref wave_clus c, arm python c
S_gt_py = fullfile(root, sprintf('validation_gt_vs_%s_class%d', py, g));          % ref GT c, arm python c
S_gt_wc = fullfile(wcf, sprintf('validation_gt_vs_wave_clus_notch_class%d', g));  % ref GT c, arm wave_clus c
% which window for which spikes:
%   wave_clus spikes -> S_wc_py, 'gt'      python spikes -> S_wc_py, 'arm' (or S_gt_py, 'arm': same python class)
%   GT spikes        -> S_gt_py, 'gt'      (S_gt_wc 'gt' is the same GT table)
% If a run is missing, make it (these are what build_ppt_report / ppt_gt_subsets call):
% python_cluster_validation(session, 'wave_clus notch', py_c, 'python', py, 'stages', {'class_match', 'rasters'}, 'out_dir', S_wc_py);
% c = cfg; c.out_dir = S_gt_py; c.stages = {'class_match', 'rasters'};
% c.arms = struct('name', 'python_sd1', 'dir', fullfile(root, py), 'times_file', tf, 'grapes_file', 'grapes_blanks_rescue.mat', 'class', []);
% run_validation(c);
% Window rule (run_validation): latency / duration window; stimuli with latency > cfg.max_lat (400 ms) or duration
% > cfg.max_dura (250 ms) also get a default-window version ([tmin_median tmax_median], 50-550 ms) -> 'window','default'.

%% 4  define subsets -- EDIT HERE ------------------------------------------------------------------------------
% each row: name | spike times (ms) | waveforms | color | raster state | side
% (masks are logical vectors on the tables of section 2)
in_wc = W.class == wc_c; in_py = P.class == py_c; in_gt = G.class == g;
wc_only = in_wc & (isnan(W.python_class) | (W.python_class ~= py_c & ~W.python_quarantined));   % venn: wave_clus only
wc_q    = in_wc & W.python_quarantined & W.python_class ~= py_c;                                  % wave_clus x quarantine
py_only = in_py & ~P.quarantined & (isnan(P.wc_notch_class) | P.wc_notch_class ~= wc_c);          % venn: python only
q_only  = P.quarantined & ~in_py & (isnan(P.wc_notch_class) | P.wc_notch_class ~= wc_c);          % quarantine only
gt_in_py2 = in_gt & G.python_class == 2;                      % example: GT c spikes in python c2
sets = {'wave_clus_only',      W.t(wc_only), M.waves.wc_notch(wc_only, :), [0 .45 .7],  S_wc_py, 'gt'; ...
        'wave_clus_x_q',       W.t(wc_q),    M.waves.wc_notch(wc_q, :),    [.8 .15 .15], S_wc_py, 'gt'; ...
        'python_only',         P.t(py_only), M.waves.python(py_only, :),   [.9 .6 0],   S_wc_py, 'arm'; ...
        'python_q_only',       P.t(q_only),  M.waves.python(q_only, :),    [.8 .15 .15], S_wc_py, 'arm'; ...
        'gt_in_python_c2',     G.t(gt_in_py2), M.waves.gt(gt_in_py2, :),   [0 0 0],     S_gt_py, 'gt'};
% a quarantine property instead, e.g. only 'low amplitude' among python quarantine-only:
% m = q_only & P.q_low_amplitude;  sets(end+1, :) = {'q_only_low_amp', P.t(m), M.waves.python(m, :), [.6 0 0], S_wc_py, 'arm'};
% P.Properties.VariableNames(startsWith(P.Properties.VariableNames, 'q_'))   % the property flags available
for i = 1:size(sets, 1), fprintf('%-20s %7d spikes\n', sets{i, 1}, numel(sets{i, 2})); end

%% 5  waveforms: one panel per subset, with the class templates on top -----------------------------------------
tm = struct('name', {sprintf('GT c%d', g), sprintf('wave_clus notch c%d', wc_c), sprintf('python c%d', py_c)}, ...
    'wave', {M.templates.gt(g, :), M.templates.wc_notch(wc_c, :), M.templates.python(py_c, :)}, ...
    'color', {[0 0 0], [0 .45 .7], [.9 .6 0]}, 'style', {'-', '--', ':'});
grp = struct('name', strrep(sets(:, 1)', '_', ' '), 'W', sets(:, 3)', 'color', sets(:, 4)');
plot_subset_waveforms(grp, tm, sprintf('ch%d: my subsets', ch), fullfile(OUT, 'waveforms_my_subsets.png'), 'ncol', 3);
% (up to 5000 single spikes per panel + mean; 'ylim', [lo hi] to fix the axis)

%% 6  rasters of each subset, with the full-run windows of its own sort ----------------------------------------
for i = 1:size(sets, 1)
    [nm, t, ~, ~, st, side] = sets{i, :};
    if isempty(t), fprintf('%s: no spikes\n', nm); continue; end
    plot_subset_rasters([], 'dir', st, 'times', t, 'name', nm, 'side', side, 'out_dir', OUT);
    w = load(fullfile(st, 'validation_state.mat'), 'ref_tables_default');
    if isfield(w, 'ref_tables_default') && ~isempty(w.ref_tables_default)       % late latency / long duration
        plot_subset_rasters([], 'dir', st, 'times', t, 'name', [nm '_default_window'], 'side', side, ...
            'out_dir', OUT, 'window', 'default');
    end
end
% pages: OUT/<name>/..._resp_window_subscr*_fig*.png ; title = in / out of the green windows; footer = which windows

%% 7  rasters of a whole class with ITS OWN windows (e.g. a class that is in no comparison) --------------------
% plot_class_rasters(S_gt_py, fullfile(root, 'hand selected'), 'grapes_blanks.mat', tf, 2, 'gt_c2', OUT);
% plot_class_rasters(S_gt_py, fullfile(root, py), 'grapes_blanks_rescue.mat', tf, 3, 'python_c3', OUT);
% plot_class_rasters(S_gt_py, wcf, 'grapes_blanks.mat', tf, 2, 'wave_clus_c2', OUT);

%% 8  bookkeeping: venn regions = subsets, regions sum to each class, in-window sums = the raster pages ---------
ok = ppt_check_unit(session, multi);   % writes <for ppt>/accounting.csv
type(fullfile(rep, 'accounting.csv'))
% quick in-window count of any times with a run's windows:
% st = load_validation_state(S_wc_py); T = st.ref_tables{1};      % {1} = reference (gt side), {2} = arm
% n_in = count_in_windows(W.t(wc_only), T.stim_number, T.win_lo, T.win_hi, st.onset_map)
