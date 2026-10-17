function venn_wave_clus(session, wc_folder, varargin)
% VENN_WAVE_CLUS  The class-match + venn comparisons for one GT unit and one wave_clus run:
%   GT vs wave_clus      -> <channel>/<wc_folder>/venn_gt_vs_wave_clus_class<g>
%   wave_clus vs python  -> <channel>/<wc_folder>/venn_wave_clus_vs_python_c<X>
%   GT vs python         -> <channel>/venn_gt_vs_python_class<g>   (only with 'gt_vs_python', true)
%
%   venn_wave_clus('pt6emu13_ch262_class3', 'wave_clus no notch')
%   venn_wave_clus('pt6emu13_ch262_class3', 'wave_clus', 'gt_vs_python', true, 'python', 'python_recluster_20261014_second_pass')
%
% SESSION   an id from validation_sessions (gives the channel folder, the GT times file / class and
%           the stimulus list); validation_sessions() lists them.
% WC_FOLDER the wave_clus run's folder inside the channel folder. It needs the times file (same name
%           as the GT's) and grapes_blanks.mat (build_wave_clus_arm(root, ch, wc_dir) makes it).
% Name-value:
%   'python'        python arm folder (default: newest python_recluster_*_second_pass in the channel)
%   'gt_vs_python'  also (re)make the GT vs python comparison (default false)
%   'tol'           matching tolerance in ms (default: the session's, 0.5)
%
% Python's class X = the python cluster that matches the GT class. The wave_clus vs python
% comparison is built around X (python_cluster_validation): python stays the arm, so its
% quarantined / rescued spikes show in the venn, and the wave_clus class is the one holding most
% of python cluster X's spikes. Each output folder holds the two class-match figures
% (class_match_*.png, both directions) and venn_three_set/ (cluster, full and all-detections venns).
p = inputParser;
addParameter(p, 'python', '');
addParameter(p, 'gt_vs_python', false);
addParameter(p, 'tol', []);
parse(p, varargin{:}); o = p.Results;
here = fileparts(mfilename('fullpath'));
addpath(here, fullfile(fileparts(here), 'real_response_analysis'), fullfile(fileparts(here), 'responses'));

cfg = validation_sessions(session); root = cfg.root; tf = cfg.gt.times_file; g = cfg.gt.class;
if ~isempty(o.tol), cfg.tol = o.tol; end
wc = fullfile(root, wc_folder);
assert(exist(fullfile(wc, tf), 'file') == 2, 'venn_wave_clus: no %s in %s', tf, wc);
assert(exist(fullfile(wc, 'grapes_blanks.mat'), 'file') == 2, 'venn_wave_clus: no grapes_blanks.mat in %s (build_wave_clus_arm)', wc);
if isempty(o.python)
    d = dir(fullfile(root, 'python_recluster_*_second_pass')); o.python = d(end).name;
end
base = cfg; base.stages = {'class_match', 'venn'}; base.venn_detections = true;

% GT vs wave_clus
c = base; c.arms = struct('name', 'wave_clus', 'dir', wc, 'times_file', tf, 'grapes_file', 'grapes_blanks.mat', 'class', []);
c.out_dir = fullfile(wc, sprintf('venn_gt_vs_wave_clus_class%d', g)); run_validation(c);

% GT vs python: kept in the channel folder, or only matched in a temp folder to find python's class X
c = base; c.arms = struct('name', 'python_sd1', 'dir', fullfile(root, o.python), 'times_file', tf, ...
                          'grapes_file', 'grapes_blanks_rescue.mat', 'class', []);
if o.gt_vs_python
    c.out_dir = fullfile(root, sprintf('venn_gt_vs_python_class%d', g));
else
    c.out_dir = fullfile(tempdir, 'venn_wave_clus_gt_vs_python'); c.stages = {'class_match'};
end
run_validation(c);
A = readtable(fullfile(c.out_dir, 'arm_classes.csv')); X = A.class(1);

% wave_clus vs python, around python cluster X
python_cluster_validation(session, wc_folder, X, 'python', o.python, 'tol', cfg.tol, ...
    'out_dir', fullfile(wc, sprintf('venn_wave_clus_vs_python_c%d', X)));
fprintf('[venn_wave_clus] %s / %s: done (GT class %d, python %s c%d)\n', session, wc_folder, g, o.python, X);
end
