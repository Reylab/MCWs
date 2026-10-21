SC = '/mnt/data0/MCWs Sessions/matlab validation/single channel';
targets = {'pt 11 emu 17 ch 323', 'pt 12 emu 14 ch 300', 'pt 13 emu 11 ch 290'};
sess = {'/mnt/data0/sEEG_DATA/MCW-FH_011/EMU/EMU-017_subj-MCW-FH_011_task-RSVPDynamicScr_run-03', ...
        dir('/mnt/data0/sEEG_DATA/MCW-FH_012/EMU/EMU-014_*'), dir('/mnt/data0/sEEG_DATA/MCW-FH_013/EMU/EMU-011_*')};
for i = 1:numel(targets)
    A = load(fullfile(SC, targets{i}, 'finalevents.mat'));
    fprintf('\n== %s (has field: %d)\n', targets{i}, isfield(A, 'seq_beg_blanks_cell'));
    tok = regexp(targets{i}, 'pt (\d+) emu (\d+)', 'tokens', 'once');
    cand = dir(fullfile(SC, sprintf('*pt %s emu %s ch*', tok{1}, tok{2})));
    files = arrayfun(@(c) fullfile(SC, c.name, 'finalevents.mat'), cand, 'UniformOutput', false);
    s = sess{i};
    if isstruct(s), s = arrayfun(@(x) fullfile(x.folder, x.name), s(~contains({s.name}, 'BAD')), 'UniformOutput', false); else, s = {s}; end
    for k = 1:numel(s)
        e = dir(fullfile(s{k}, '*finalevents*.mat'));
        files = [files; fullfile({e.folder}, {e.name})']; %#ok<AGROW>
    end
    for k = 1:numel(files)
        if ~exist(files{k}, 'file'), continue; end
        B = load(files{k});
        same = isequaln(A.pics_onset, B.pics_onset);
        fprintf('  %-130s field=%d same_onsets=%d\n', strrep(files{k}, '/mnt/data0/', ''), isfield(B, 'seq_beg_blanks_cell'), same);
    end
end
