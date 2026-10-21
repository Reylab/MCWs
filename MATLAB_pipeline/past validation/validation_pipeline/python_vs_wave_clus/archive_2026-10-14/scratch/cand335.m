A = '/mnt/data0/MCWs Sessions/matlab validation/analysis trials old/new trial pt 9 emu 3 ch 335';
F = {'/mnt/data0/sEEG_DATA/MCW-FH_009/EMU/EMU-003_subj-MCW-FH_009_task-RSVPDynamicScr_run-01/times_mLFUS06 raw_335_original.mat', ...
     '/mnt/data0/sEEG_DATA/MCW-FH_009/EMU/EMU-003_subj-MCW-FH_009_task-RSVPDynamicScr_run-01/times_mLFUS06 raw_335.mat', ...
     '/mnt/data0/MCWs Sessions/matlab validation/single channel/zneed rerun grapes blanks changednew trial pt 9 emu 3 ch 335/hand selected/newer_times.mat', ...
     fullfile(A, 'hand selected', 'times_mLFUS06 raw_335.mat'), fullfile(A, 'hand selected', 'newer_times.mat'), ...
     fullfile(A, 'RECLUSTER_VALIDATION_20260910_155543', 'handselect', 'times_mLFUS06 raw_335.mat'), ...
     fullfile(A, 'RECLUSTER_VALIDATION_20260910_155543', 'handselect', 'reconstructed_cc_mLFUS06 raw_335.mat')};
want = [14960 8159 6083 2423 2565 3245];
for i = 1:numel(F)
    w = whos('-file', F{i});
    d = dir(F{i});
    if ~any(strcmp({w.name}, 'cluster_class'))
        fprintf('%s (%s): no cluster_class [%s]\n', strrep(F{i}, '/mnt/data0/', ''), d.date, strjoin({w.name}, ' '));
        continue
    end
    T = load(F{i}, 'cluster_class');
    n = arrayfun(@(c) nnz(T.cluster_class(:,1) == c), 0:max(T.cluster_class(:,1)));
    tag = '';
    if numel(n) == 7 && isequal(n(2:end), want), tag = '<-- MATCHES GRAPES EXACTLY'; elseif numel(n) > 2 && n(3) == 8159, tag = '<-- class 2 = 8159'; end
    fprintf('%s (%s)\n   class 0..: %s %s\n', strrep(F{i}, '/mnt/data0/', ''), d.date, mat2str(n), tag);
end
% grapes-type files for ch335 in those folders
G = [dir(fullfile(A, '**', 'grapes*.mat')); dir('/mnt/data0/sEEG_DATA/MCW-FH_009/EMU/EMU-003_subj-MCW-FH_009_task-RSVPDynamicScr_run-01/*grapes*.mat'); ...
     dir('/mnt/data0/MCWs Sessions/matlab validation/single channel/zneed rerun grapes blanks changednew trial pt 9 emu 3 ch 335/hand selected/grapes*.mat')];
fprintf('\ngrapes files (chan335 classN counts):\n');
for g = G'
    f = fullfile(g.folder, g.name);
    try
        R = load(f, 'rasters');
        if ~isfield(R.rasters, 'chan335'), continue; end
        dd = R.rasters.chan335.details; fn = fieldnames(dd); fn = fn(~cellfun(@isempty, regexp(fn, '^class\d+$', 'once')));
        v = cellfun(@(x) dd.(x), fn)';
        fprintf('  %s (%s): %s\n', strrep(f, '/mnt/data0/', ''), g.date, mat2str(v));
    catch
    end
end
