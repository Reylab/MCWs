function T = run_wave_clus_validations(session, wc_dirs, varargin)
% RUN_WAVE_CLUS_VALIDATIONS  For every wave_clus run of one channel, two
% validation runs, written INTO that wave_clus folder:
%   1  GT vs wave_clusN                     -> <wave_clusN>/validation_gt_vs_<wave_clusN>[_classK]
%   2  wave_clusN (the class matched to GT in 1) as the reference vs the
%      python recluster arm                 -> <wave_clusN>/validation_<wave_clusN>_vs_python_sd1[_classK]
%   3  python's GT cluster as the reference vs wave_clusN as the arm (the
%      wave_clus cluster matched to python's GT cluster): the comparison
%      measured from python's side -> <wave_clusN>/validation_python_sd1_vs_<wave_clusN>[_classK]
%      (optional, 'steps' with 'py_ref'; not used by default -- python has to be
%      the arm so its quarantine / rescue / Mahalanobis breakdown is analysed)
% Both sides are the clusters closest to the GT unit: the wave_clus class
% matched to GT in 1, and the python class matched to GT (same 0.5 ms spike
% matching as GT vs python) -- never a python class matched through the
% wave_clus cluster, which follows wave_clus's merges instead of the unit.
%
%   run_wave_clus_validations('pt6_ch333', {'wave_clus1','wave_clus2'})
%   run_wave_clus_validations('pt12_ch300_class2', 'wave_clus1', 'python', 'python_recluster_20261012_second_pass')
%
% session  - validation_sessions id: channel folder, stim list, GT unit
%            (_classK in the id is carried into the output folder names)
% wc_dirs  - wave_clus run folder(s) under the channel folder; each needs its
%            grapes_blanks.mat (build_wave_clus_arm(root, ch, wc_dir) in place)
% Name-value: 'redo_gt' (default true; false reuses a finished step 1),
%             'python' arm folder (default 'python_recluster_20261012_second_pass',
%             grapes_blanks_rescue.mat), 'stages' (default 'all')
% Returns one row per wave_clus run: matched wave_clus class and both output folders.

p = inputParser;
addParameter(p, 'python', 'python_recluster_20261012_second_pass');
addParameter(p, 'stages', 'all');
addParameter(p, 'redo_gt', true);
addParameter(p, 'python_class', []);   % fix python's class (default: python's cluster matched to GT)
addParameter(p, 'py_tag', '');         % suffix for the python comparison's folder (e.g. '_pyc1')
addParameter(p, 'wc_class', []);       % fix the wave_clus reference class (default: matched to GT)
addParameter(p, 'steps', {'gt', 'wc_ref'});   % any of: gt (1), wc_ref (2), py_ref (3; off: python must stay the arm)   % false: reuse a finished GT-vs-wave_clus run (its arm_classes.csv)
parse(p, varargin{:});
o = p.Results;
here = fileparts(mfilename('fullpath'));
addpath(here, fullfile(fileparts(here), 'real_response_analysis'), fullfile(fileparts(here), 'responses'));

tok = regexp(session, '_class\d+$', 'match', 'once');   % '' or '_classK'
wc_dirs = cellstr(wc_dirs);
% python class closest to the GT unit
b0 = validation_sessions(session);
G = load(fullfile(b0.gt.dir, b0.gt.times_file), 'cluster_class');
gt_t = sort(G.cluster_class(G.cluster_class(:, 1) == b0.gt.class, 2));
P = load(fullfile(b0.root, o.python, b0.gt.times_file), 'cluster_class');
py_class = match_arms_to_ground_truth([repmat(b0.gt.class, numel(gt_t), 1) gt_t], gt_t, {'python_sd1'}, ...
    {P.cluster_class}, b0.ch, b0.tol, b0.min_recall);
fprintf('[wave_clus validations] %s: python class %d is closest to GT class %d\n', session, py_class, b0.gt.class);
if ~isempty(o.python_class)
    py_class = o.python_class;
    fprintf('[wave_clus validations] %s: python class fixed to %d\n', session, py_class);
end
rows = cell(numel(wc_dirs), 4);
for i = 1:numel(wc_dirs)
    wc = wc_dirs{i};
    base = validation_sessions(session);
    tf = base.gt.times_file;

    % 1: GT vs wave_clus (wave_clus class matched to GT by spike time)
    c1 = base;
    c1.arms = struct('name', wc, 'dir', fullfile(base.root, wc), 'times_file', tf, ...
        'grapes_file', 'grapes_blanks.mat', 'class', []);
    c1.out_dir = fullfile(base.root, wc, ['validation_gt_vs_' wc tok]);
    c1.venn_detections = true; c1.channel_subsets = true; c1.stages = o.stages;
    chan_level = fullfile(base.root, ['validation_gt_vs_' wc tok]);   % a finished copy at the channel level
    if ~o.redo_gt && ~exist(fullfile(c1.out_dir, 'arm_classes.csv'), 'file') && exist(fullfile(chan_level, 'arm_classes.csv'), 'file')
        c1.out_dir = chan_level;
    end
    if ismember('gt', o.steps) && (o.redo_gt || ~exist(fullfile(c1.out_dir, 'arm_classes.csv'), 'file'))
        run_validation(c1);
    end
    m = readtable(fullfile(c1.out_dir, 'arm_classes.csv'), 'TextType', 'char');
    wc_class = m.class(1);
    if ~isempty(o.wc_class), wc_class = o.wc_class; end

    % 2: that wave_clus class as the reference vs the python recluster arm
    c2 = base;
    c2.gt = struct('name', wc, 'dir', fullfile(base.root, wc), 'times_file', tf, ...
        'grapes_file', 'grapes_blanks.mat', 'class', wc_class);
    c2.arms = struct('name', 'python_sd1', 'dir', fullfile(base.root, o.python), 'times_file', tf, ...
        'grapes_file', 'grapes_blanks_rescue.mat', 'class', py_class);
    c2.out_dir = fullfile(base.root, wc, sprintf('validation_%s_vs_python_sd1%s%s', wc, tok, o.py_tag));
    c2.venn_detections = true; c2.channel_subsets = true; c2.stages = o.stages;
    if ismember('wc_ref', o.steps), run_validation(c2); end

    % 3: python's GT cluster as the reference, this wave_clus run as the arm
    c3 = base;
    c3.gt = struct('name', 'python_sd1', 'dir', fullfile(base.root, o.python), 'times_file', tf, ...
        'grapes_file', 'grapes_blanks_rescue.mat', 'class', py_class);
    c3.arms = struct('name', wc, 'dir', fullfile(base.root, wc), 'times_file', tf, ...
        'grapes_file', 'grapes_blanks.mat', 'class', []);
    c3.out_dir = fullfile(base.root, wc, sprintf('validation_python_sd1_vs_%s%s', wc, tok));
    c3.venn_detections = true; c3.channel_subsets = true; c3.stages = o.stages;
    if ismember('py_ref', o.steps), run_validation(c3); end
    rows(i, :) = {wc, wc_class, c1.out_dir, c2.out_dir};
    fprintf('[wave_clus validations] %s %s: GT class %d -> %s class %d\n', session, wc, base.gt.class, wc, wc_class);
end
T = cell2table(rows, 'VariableNames', {'wave_clus', 'wave_clus_class', 'gt_vs_wave_clus', 'wave_clus_vs_python'});
end
