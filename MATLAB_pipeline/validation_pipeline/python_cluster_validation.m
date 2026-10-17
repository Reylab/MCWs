function cfg = python_cluster_validation(session, wc_run, X, varargin)
% PYTHON_CLUSTER_VALIDATION  wave_clus vs python built around python cluster X:
% python stays the ARM (so its quarantined / rescued spikes are in the venns and
% composition) with its class fixed to X, and the wave_clus reference class is the
% wave_clus cluster holding the most of python cluster X's spikes (one-to-one,
% cfg.tol). Runs class_match + venn (incl. the all-detections venn) by default.
%
%   python_cluster_validation('pt4_ch259', 'wave_clus3', 1)
%   python_cluster_validation('pt4_ch259', 'wave_clus3', 1, 'python', 'python_recluster_20261012_second_pass', ...
%                             'stages', {'class_match','venn','composition'}, 'out_dir', '/some/folder')
%
% Name-value:
%   'python'   python arm folder under the session root (default: newest python_recluster_*_second_pass)
%   'stages'   run_validation stages, default {'class_match','venn'}
%   'out_dir'  default <root>/<wc_run>/validation_<wc_run>_vs_python_c<X>
%   'tol'      matching tolerance in ms (default: the session's, 0.5)
p = inputParser;
addParameter(p, 'python', '');
addParameter(p, 'stages', {'class_match', 'venn'});
addParameter(p, 'out_dir', '');
addParameter(p, 'tol', []);
parse(p, varargin{:});
o = p.Results;
here = fileparts(mfilename('fullpath'));
addpath(here, fullfile(fileparts(here), 'real_response_analysis'), fullfile(fileparts(here), 'responses'));

cfg = validation_sessions(session);
if ~isempty(o.tol), cfg.tol = o.tol; end
root = cfg.root; tf = cfg.gt.times_file;
if isempty(o.python)
    d = dir(fullfile(root, 'python_recluster_*_second_pass'));
    assert(~isempty(d), 'python_cluster_validation: no python_recluster_*_second_pass folder in %s', root);
    o.python = d(end).name;
end
P = load(fullfile(root, o.python, tf), 'cluster_class');
W = load(fullfile(root, wc_run, tf), 'cluster_class');
px = sort(P.cluster_class(P.cluster_class(:, 1) == X, 2));
assert(~isempty(px), 'python_cluster_validation: python cluster %d has no spikes in %s (classes: %s)', ...
    X, o.python, mat2str(unique(P.cluster_class(:, 1))'));
wc_class = match_arms_to_ground_truth([repmat(X, numel(px), 1) px], px, {wc_run}, {W.cluster_class}, ...
    cfg.ch, cfg.tol, cfg.min_recall);
fprintf('[python_cluster_validation] python %s c%d (%d spikes) -> %s c%d\n', o.python, X, numel(px), wc_run, wc_class);

cfg.gt = struct('name', wc_run, 'dir', fullfile(root, wc_run), 'times_file', tf, ...
                'grapes_file', 'grapes_blanks.mat', 'class', wc_class);
cfg.arms = struct('name', 'python_sd1', 'dir', fullfile(root, o.python), 'times_file', tf, ...
                  'grapes_file', 'grapes_blanks_rescue.mat', 'class', X);
if isempty(o.out_dir), o.out_dir = fullfile(root, wc_run, sprintf('validation_%s_vs_python_c%d', wc_run, X)); end
cfg.out_dir = o.out_dir;
cfg.stages = o.stages;
cfg.venn_detections = true;
run_validation(cfg);
end
