function cmp_events()
SC = '/mnt/data0/MCWs Sessions/matlab validation/single channel';
F = {fullfile(SC, 'pt 11 emu 17 ch 321', 'finalevents.mat'), fullfile(SC, 'pt 11 emu 17 ch 328', 'finalevents.mat'), ...
     '/mnt/data0/sEEG_DATA/MCW-FH_011/EMU/EMU-017_subj-MCW-FH_011_task-RSVPDynamicScr_run-03/finalevents.mat'};
for i = 1:numel(F)
    d = dir(F{i}); w = whos('-file', F{i}); L = load(F{i});
    fprintf('%s\n   [%s] vars: %s\n', F{i}, d.date, strjoin({w.name}, ' '));
end
A = load(F{1}); B = load(F{2}); C = load(F{3});
fprintf('pics_onset ch321 == ch328: %d ; ch328 == sEEG: %d\n', isequaln(A.pics_onset, B.pics_onset), isequaln(B.pics_onset, C.pics_onset));
fprintf('seq_beg_blanks_cell ch328 == sEEG: %d\n', isequaln(B.seq_beg_blanks_cell, C.seq_beg_blanks_cell));
common = intersect(fieldnames(A), fieldnames(B));
for k = common', fprintf('   %s: ch321 == ch328 %d\n', k{1}, isequaln(A.(k{1}), B.(k{1}))); end
end
