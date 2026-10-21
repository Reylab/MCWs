function ok = check_arm_grapes(chan_folder, arm_subdir, ch)
% CHECK_ARM_GRAPES  For every python arm python_recluster_* in <chan_folder>/<arm_subdir>: the number of spikes of
% each class in the arm's times_*.mat equals rasters.chan<ch>.details.class<k> in its grapes_blanks.mat and
% grapes_blanks_rescue.mat. Prints one line per arm and grapes file; ok = all match.
%   chan_folder  absolute channel folder, or a folder name under the single channel root (mcws_paths)
%   arm_subdir   '' (arms in the channel folder) or e.g. 'run isi3_sd1'
%   check_arm_grapes('/data/pt 6 emu 13 ch 264', '', 264)
if ~startsWith(chan_folder, filesep)
    chan_folder = fullfile(mcws_paths().single_channel, chan_folder);
end
d = dir(fullfile(chan_folder, arm_subdir, 'python_recluster_*')); d = d([d.isdir]); ok = ~isempty(d);
for a = d'
    t = dir(fullfile(a.folder, a.name, 'times_*.mat'));
    c = load(fullfile(t(1).folder, t(1).name), 'cluster_class'); c = c.cluster_class(:, 1); k = unique(c(c > 0))';
    tc = arrayfun(@(x) nnz(c == x), k);
    for gname = {'grapes_blanks.mat', 'grapes_blanks_rescue.mat'}
        G = load(fullfile(a.folder, a.name, gname{1}), 'rasters'); dt = G.rasters.(sprintf('chan%d', ch)).details;
        gc = arrayfun(@(x) dt.(sprintf('class%d', x)), k);
        m = isequal(tc, gc); ok = ok && m; lab = 'DIFF'; if m, lab = 'MATCH'; end
        fprintf('[grapes] %s / %s / %s: %s  times %s  grapes %s\n', chan_folder, a.name, gname{1}, lab, mat2str(tc), mat2str(gc));
    end
end
end
