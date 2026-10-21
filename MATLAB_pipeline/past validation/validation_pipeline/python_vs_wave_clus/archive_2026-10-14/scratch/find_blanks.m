function find_blanks()
SC = '/mnt/data0/MCWs Sessions/matlab validation/single channel';
for d = {'pt 6 emu 7 ch 334', 'pt 9 emu 3 ch 337', 'pt 10 emu 29 ch 291', 'pt 11 emu 17 ch 321', 'pt 14 emu 7 ch 363'}
    fprintf('%-22s %s\n', d{1}, holders(fullfile(SC, d{1})));
end
for d = {'MCW-FH_006/EMU/EMU-007_subj-MCW-FH_006_task-RSVPdynamic_scr_run-02', 'MCW-FH_010/EMU/EMU-029_subj-MCW-FH_010_task-RSVPCategLocaliz_run-01', ...
         'MCW-FH_011/EMU/EMU-017_subj-MCW-FH_011_task-RSVPDynamicScr_run-03', 'MCW-FH_014/EMU/EMU-007_subj-MCW-FH_014_task-RSVPCategLocaliz_run-01'}
    fprintf('sEEG %-14s %s\n', d{1}(1:10), holders(fullfile('/mnt/data0/sEEG_DATA', d{1})));
end
end
function s = holders(d)
F = dir(fullfile(d, '*.mat')); hit = {};
for f = F'
    try
        w = whos('-file', fullfile(f.folder, f.name));
        if any(strcmp({w.name}, 'seq_beg_blanks_cell')), hit{end+1} = sprintf('%s [%s]', f.name, f.date); end %#ok<AGROW>
    catch
    end
end
s = strjoin(hit, ', '); if isempty(s), s = '(none)'; end
end
