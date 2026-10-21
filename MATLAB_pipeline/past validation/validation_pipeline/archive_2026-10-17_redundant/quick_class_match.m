function T = quick_class_match(session, arm_dirs, varargin)
% QUICK_CLASS_MATCH  Only the class_match figure (GT class vs arm clusters),
% straight from times files -- no grapes, no full validation run.
%
%   quick_class_match('pt12_ch299', 'wave_clus_20261012')
%   quick_class_match('pt12_ch299', {'wave_clus_20261012', 'python_recluster_20261012_second_pass'})
%   quick_class_match('/mnt/.../single channel/pt 12 emu 18 ch 299', 'wave_clus_20261012', 'gt_class', 1)
%
% session  - a validation_sessions id (GT folder, times file and class come
%            from its row) or a channel folder (GT = '<folder>/hand selected',
%            class from 'gt_class', default 1)
% arm_dirs - arm subfolder name(s) under the channel folder; each must hold
%            the channel's times_<name>.mat (wave_clus, python, ...)
%
% Name-value: 'gt_class' (override), 'out_dir' (default <folder>/class_match),
%             'tol' ms (0.5), 'min_recall' (0.5),
%             'ref' subfolder to use as the reference instead of GT (e.g. a
%             wave_clus run), with 'gt_class' = its class:
%   quick_class_match('pt12_ch299', 'python_recluster_20261012_second_pass', 'ref', 'wave_clus7', 'gt_class', 1)
%             'both' (default true): also the other way round -- each arm's
%             matched cluster as the reference over the reference's classes
% Writes class_match_ch<ch>_<reference>_vs_<arm>.png per arm (and, with
% 'both', class_match_ch<ch>_<arm>_vs_<reference>.png); returns the matched
% class per arm.

p = inputParser;
addParameter(p, 'gt_class', []);
addParameter(p, 'out_dir', '');
addParameter(p, 'tol', 0.5);
addParameter(p, 'min_recall', 0.5);
addParameter(p, 'ref', '');
addParameter(p, 'both', true);
parse(p, varargin{:});
o = p.Results;

here = fileparts(mfilename('fullpath'));
P = fileparts(here);
addpath(here, fullfile(P, 'real_response_analysis'), fullfile(P, 'responses'));

if exist(session, 'dir')
    root = session;
    gt_dir = fullfile(root, 'hand selected');
    tf = dir(fullfile(gt_dir, 'times_*.mat'));
    assert(numel(tf) == 1, 'expected one times_*.mat in %s, found %d', gt_dir, numel(tf));
    times_file = tf.name;
    gt_class = 1;
    tok = regexp(times_file, '_(\d+)\.mat$', 'tokens', 'once');
    ch = str2double(tok{1});
else
    cfg = validation_sessions(session);
    root = cfg.root; gt_dir = cfg.gt.dir; times_file = cfg.gt.times_file;
    gt_class = cfg.gt.class; ch = cfg.ch;
end
if ~isempty(o.ref), gt_dir = fullfile(root, o.ref); end
if ~isempty(o.gt_class), gt_class = o.gt_class; end
arm_dirs = cellstr(arm_dirs);
out = o.out_dir;
if isempty(out), out = fullfile(root, 'class_match'); end
if ~exist(out, 'dir'), mkdir(out); end

gt = load(fullfile(gt_dir, times_file), 'cluster_class');
gt_t = sort(gt.cluster_class(gt.cluster_class(:, 1) == gt_class, 2));
assert(~isempty(gt_t), 'GT class %d has no spikes in %s', gt_class, fullfile(gt_dir, times_file));
cc = cell(1, numel(arm_dirs));
for k = 1:numel(arm_dirs)
    a = load(fullfile(root, arm_dirs{k}, times_file), 'cluster_class');
    cc{k} = a.cluster_class;
end
[cls, ~, DET, COMP] = match_arms_to_ground_truth([repmat(gt_class, numel(gt_t), 1) gt_t], gt_t, ...
    arm_dirs, cc, ch, o.tol, o.min_recall);
ref = 'GT'; if ~isempty(o.ref), ref = o.ref; end
class_match_figure(DET, COMP, ch, arm_dirs, out, ref);
if o.both
    for k = 1:numel(arm_dirs)
        at = sort(cc{k}(cc{k}(:, 1) == cls(k), 2));
        if isnan(cls(k)) || isempty(at), continue; end
        [~, ~, Dr, Cr] = match_arms_to_ground_truth([repmat(cls(k), numel(at), 1) at], at, ...
            {ref}, {gt.cluster_class}, ch, o.tol, o.min_recall);
        class_match_figure(Dr, Cr, ch, {ref}, out, arm_dirs{k});
    end
end
T = table(arm_dirs(:), cls(:), 'VariableNames', {'arm', 'matched_class'});
fprintf('[class_match] ch%d %s class %d -> %s\n', ch, ref, gt_class, out);
end
