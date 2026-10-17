function cfg = rerun_validation(stages, varargin)
% RERUN_VALIDATION  Redo stages of a finished run_validation from its output
% folder (validation_state.mat holds the cfg), optionally with a different
% arm class or GT class -- no config file needed.
%
%   cd <validation out_dir, or any folder under it>
%   rerun_validation('venn')                          % redo one stage, same classes
%   rerun_validation({'venn','composition'})
%   rerun_validation('all', 'class', 7)               % arm 1 forced to class 7
%   rerun_validation('venn', 'arm', 'python_sd2', 'class', 7)
%   rerun_validation('class_match', 'gt_class', 2)    % match the arms against GT class 2
%   rerun_validation('subset_rasters', 'subsets', {'gt_only','arm_only'})
%
% Stages: 'class_match' 'rasters' 'venn' 'composition' 'pie_venn' 'waveforms'
%         'subsets' 'subset_rasters' 'results' 'all' (= every run_validation stage
%         except subset_rasters, which runs only when named)
%
% Name-value:
%   'dir'       validation out_dir (or a folder under it), default pwd
%   'arm'       arm the 'class' override applies to (name or index), default 1
%   'class'     force that arm's class (instead of the GT-matched one)
%   'gt_class'  use this GT class instead of cfg.gt.class (arms are re-matched
%               against it unless 'class' is also given)
%   'subsets'   for 'subset_rasters': subset names, 'default' (gt_only,
%               arm_only, matched, q_not_rescued, qnr_<property>) or 'all'
%   'out_dir'   where to write; default: the run's own folder when no class is
%               overridden, else <run>/reruns/<arm>_class<c>[_gt<g>] so the
%               original pages are not overwritten
%
% Returns the cfg it ran with (so you can tweak it and call run_validation(cfg)).

p = inputParser;
addParameter(p, 'dir', pwd);
addParameter(p, 'arm', 1);
addParameter(p, 'class', []);
addParameter(p, 'gt_class', []);
addParameter(p, 'subsets', 'default');
addParameter(p, 'out_dir', '');
parse(p, varargin{:});
o = p.Results;

st = load_validation_state(o.dir);
cfg = st.cfg;
stages = cellstr(stages);

overridden = ~isempty(o.class) || ~isempty(o.gt_class);
k = 1;
if ~isnumeric(o.arm)
    k = find(strcmp({cfg.arms.name}, o.arm), 1);
    assert(~isempty(k), 'unknown arm %s; arms: %s', o.arm, strjoin({cfg.arms.name}, ', '));
else
    k = o.arm;
end
tag = '';
if ~isempty(o.class)
    cfg.arms(k).class = o.class;
    tag = sprintf('%s_class%d', cfg.arms(k).name, o.class);
end
if ~isempty(o.gt_class)
    cfg.gt.class = o.gt_class;
    if isempty(tag), tag = 'matched'; end
    tag = sprintf('%s_gt%d', tag, o.gt_class);
end
if ~isempty(o.out_dir)
    cfg.out_dir = o.out_dir;
elseif overridden
    cfg.out_dir = fullfile(st.state_dir, 'reruns', tag);
end

want_sub_rasters = any(strcmp(stages, 'subset_rasters'));
run_stages = setdiff(stages, {'subset_rasters'}, 'stable');
% subset rasters need the raster tables (windows) of this out_dir's run
sf = fullfile(cfg.out_dir, 'validation_state.mat');
if want_sub_rasters && (~exist(sf, 'file') || isempty(getfield(load(sf, 'ref_tables'), 'ref_tables'))) %#ok<GFLD>
    run_stages = union(run_stages, {'rasters', 'subsets'}, 'stable');
end
if want_sub_rasters && ~exist(fullfile(cfg.out_dir, 'subsets'), 'dir')
    run_stages = union(run_stages, {'subsets'}, 'stable');
end

if ~isempty(run_stages)
    if any(strcmp(run_stages, 'all')), cfg.stages = 'all'; else, cfg.stages = run_stages; end
    fprintf('[rerun] stages %s -> %s\n', strjoin(cellstr(cfg.stages), ', '), cfg.out_dir);
    run_validation(cfg);
end
if want_sub_rasters
    plot_subset_rasters(o.subsets, 'dir', cfg.out_dir, 'arm', k);
end
end
