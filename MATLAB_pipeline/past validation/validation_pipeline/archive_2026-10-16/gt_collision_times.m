function t = gt_collision_times(root, times_file)
% GT_COLLISION_TIMES  Times (ms) of the spikes GT's own detection found and then removed as bundle
% collisions (codes_emu separate_collisions: the same spike within 0.5 ms on >= 5 channels of the
% bundle), read from GT's ORIGINAL <name>_spikes.mat in the recording folder (read only):
% index_all = detections before the step, index = after (= the hand-selected times).
% Python and wave_clus sort one channel at a time and keep these spikes; set cfg.exclude_ms to this
% to leave them out of the comparisons (the sorts themselves are not touched).
%
%   t = gt_collision_times('/mnt/.../single channel/pt 6 emu 13 ch 337', 'times_mLFUS08 raw_337.mat')
%
% Empty when GT's detection predates the collision step (no index_all).
stem = regexprep(times_file, '^times_|\.mat$', '');
[~, chf] = fileparts(root);
tok = regexp(chf, 'pt (\d+) emu (\d+)', 'tokens', 'once');
assert(~isempty(tok), 'gt_collision_times: cannot read patient / emu from %s', chf);
rec = dir(sprintf('/mnt/data0/sEEG_DATA/MCW-FH_%03d/EMU/EMU-%03d_*', str2double(tok{1}), str2double(tok{2})));
rec = rec([rec.isdir] & ~contains({rec.name}, 'BAD'));       % never the BAD copies
if numel(rec) > 1, rec = rec(contains({rec.name}, '_sunil')); end
assert(numel(rec) == 1, 'gt_collision_times: no single recording folder for %s', chf);
f = fullfile(rec.folder, rec.name, [stem '_spikes.mat']);
assert(exist(f, 'file') == 2, 'gt_collision_times: no %s', f);
w = whos('-file', f);
if ~any(strcmp({w.name}, 'index_all')), t = zeros(0, 1); return; end
S = load(f, 'index', 'index_all');
t = setdiff(S.index_all(:), S.index(:));
G = load(fullfile(root, 'hand selected', times_file), 'cluster_class');
if ~isequal(sort(S.index(:)), sort(G.cluster_class(:, 2)))
    warning('gt_collision_times: %s index is not the hand-selected GT (%d vs %d spikes) -- check it is the GT detection', ...
        f, numel(S.index), size(G.cluster_class, 1));
end
end
