P = '/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline';
addpath(genpath(fullfile(P, 'neuroshare'))); addpath(genpath(fullfile(P, 'codes_emu')));
SC = '/mnt/data0/MCWs Sessions/matlab validation/single channel';
W = '/tmp/claude-1017/-home-ajb-Documents-GitHub-MCWs-MATLAB-pipeline/1d8fd482-04b6-49a4-9526-6897f90343d6/scratchpad/blank_work';
J = {'pt 5 emu 4 ch 271 check', '/mnt/data0/sEEG_DATA/MCW-FH_005/EMU/EMU-004_subj-MCW-FH_005_task-RSVPdynamic_scr_run-01/EMU-004_subj-MCW-FH_005_task-RSVPdynamic_scr_run-01_RIP.nev'};
old = pwd; c0 = onCleanup(@() cd(old));
for i = 1:size(J, 1)
    [fold, nev] = J{i, :};
    root = fullfile(SC, fold);
    w = fullfile(W, strrep(fold, ' ', '_'));
    if exist(w, 'dir'), rmdir(w, 's'); end
    mkdir(w);
    copyfile(nev, w);
    copyfile(fullfile(root, 'finalevents.mat'), w);
    copyfile(fullfile(root, 'experiment_properties_online3.mat'), w);
    try
        cd(w);
        [~, n, e] = fileparts(nev);
        extract_blank_on_events_ripple([n e]);
        cd(old);
        A = load(fullfile(root, 'finalevents.mat')); B = load(fullfile(w, 'finalevents.mat'));
        added = setdiff(fieldnames(B), fieldnames(A));
        kept = all(cellfun(@(f) isequaln(A.(f), B.(f)), fieldnames(A)));
        nseq = cellfun(@numel, B.seq_beg_blanks_cell);
        fprintf('[%s] added: %s | original fields unchanged: %d | blank sequences per sub-screening: %s\n', fold, strjoin(added, ', '), kept, mat2str(nseq(:)'));
        assert(kept && isfield(B, 'seq_beg_blanks_cell'), 'check failed');
        if ~exist(fullfile(root, 'finalevents_orig.mat'), 'file'), copyfile(fullfile(root, 'finalevents.mat'), fullfile(root, 'finalevents_orig.mat')); end
        copyfile(fullfile(w, 'finalevents.mat'), fullfile(root, 'finalevents.mat'));
        fid = fopen(fullfile(root, 'hand selected', 'README_files_20261008.txt'), 'a');
        fprintf(fid, 'finalevents.mat: added %s with extract_blank_on_events_ripple (session .nev, run in a scratch copy); original kept as ../finalevents_orig.mat\n', strjoin(added, ', '));
        fclose(fid);
    catch ME
        cd(old); fprintf('[%s] FAILED: %s\n', fold, ME.message);
    end
end
