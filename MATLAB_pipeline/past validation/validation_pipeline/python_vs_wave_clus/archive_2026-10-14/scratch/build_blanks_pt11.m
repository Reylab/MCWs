% Build grapes_blanks.mat (use_blanks=true, circshiftblanks=false) from each
% channel's hand-selected times file, in a clean folder, check it, then
% install it in hand selected (an existing grapes_blanks.mat is moved to
% replaced_20261008/ first).
P = '/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline';
addpath(genpath(fullfile(P, 'codes_emu'))); addpath(genpath(fullfile(P, 'useful_functions')));
SC = '/mnt/data0/MCWs Sessions/matlab validation/single channel';
J = {'pt 11 emu 17 ch 323', 323, 3, 8641
     'pt 11 emu 17 ch 324', 324, 4, 6318
     'pt 11 emu 17 ch 326', 326, 1, 39101
     'pt 11 emu 17 ch 327', 327, 1, 37912};
old = pwd; c0 = onCleanup(@() cd(old));
for i = 1:size(J, 1)
    [fold, ch, cls, want] = J{i, :};
    root = fullfile(SC, fold); hs = fullfile(root, 'hand selected');
    try
        F = load(fullfile(root, 'finalevents.mat'));
        assert(isfield(F, 'seq_beg_blanks_cell'), 'finalevents has no seq_beg_blanks_cell');
        N = load(fullfile(root, 'NSx.mat'), 'NSx'); nm = N.NSx([N.NSx.chan_ID] == ch).output_name;
        tf = fullfile(hs, ['times_' nm '.mat']);
        T = load(tf, 'cluster_class'); lab = T.cluster_class(:,1);
        tc = arrayfun(@(c) nnz(lab == c), 1:max(lab));
        bdir = fullfile(hs, 'build_grapes_blanks_20261008');
        if exist(bdir, 'dir'), rmdir(bdir, 's'); end
        mkdir(bdir); copyfile(tf, bdir);
        cd(root);
        do_structure_sorted_BCM_online3(ch, true, false, false, 'times_dir', bdir, 'spike_dir', bdir);
        cd(old);
        G = load(fullfile(bdir, 'grapes_blanks.mat'), 'rasters');
        d = G.rasters.(sprintf('chan%d', ch)).details;
        gc = arrayfun(@(c) d.(sprintf('class%d', c)), 1:numel(tc));
        ok = isequal(gc, tc) && gc(cls) == want;
        fprintf('\n[%s] times %s | built grapes_blanks %s | class %d = %d (expected %d) -> %s\n', ...
            fold, mat2str(tc), mat2str(gc), cls, gc(cls), want, string(ok));
        if ok
            if exist(fullfile(hs, 'grapes_blanks.mat'), 'file')
                if ~exist(fullfile(hs, 'replaced_20261008'), 'dir'), mkdir(fullfile(hs, 'replaced_20261008')); end
                movefile(fullfile(hs, 'grapes_blanks.mat'), fullfile(hs, 'replaced_20261008', 'grapes_blanks.mat'));
                note = 'moved aside: grapes_blanks.mat (other channels / old)  (hand selected/replaced_20261008/)';
            else
                note = '';
            end
            movefile(fullfile(bdir, 'grapes_blanks.mat'), fullfile(hs, 'grapes_blanks.mat'));
            rmdir(bdir, 's');
            fid = fopen(fullfile(hs, 'README_files_20261008.txt'), 'a');
            if ~isempty(note), fprintf(fid, '%s\n', note); end
            fprintf(fid, 'built: grapes_blanks.mat from %s (do_structure_sorted_BCM_online3, use_blanks=1, circshiftblanks=0); class counts %s\n', ...
                ['times_' nm '.mat'], mat2str(gc));
            fclose(fid);
        end
    catch ME
        cd(old);
        fprintf('\n[%s] FAILED: %s\n', fold, ME.message);
    end
end
