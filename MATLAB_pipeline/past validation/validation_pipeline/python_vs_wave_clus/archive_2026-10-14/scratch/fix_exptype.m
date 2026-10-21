SC = '/mnt/data0/MCWs Sessions/matlab validation/single channel';
targets = {'pt 4 emu 7 ch 322/hand selected', 'pt 5 emu 4 ch 271/hand selected', 'pt 6 emu 13 ch 262/hand selected', ...
    'pt 6 emu 13 ch 264/hand selected', 'pt 6 emu 13 ch 337/hand selected', 'pt 11 emu 17 ch 323/hand selected', ...
    'pt 11 emu 17 ch 324/hand selected', 'pt 11 emu 17 ch 326/hand selected', 'pt 11 emu 17 ch 327/hand selected', ...
    'pt 12 emu 14 ch 300/hand selected', 'pt 13 emu 11 ch 290/hand selected', ...
    'pt 10 emu 7 ch 368/wave_clus_20261008', 'pt 11 emu 17 ch 328/wave_clus_20261008'};
for i = 1:numel(targets)
    d = fullfile(SC, targets{i}); root = fileparts(d);
    f = fullfile(d, 'grapes_blanks.mat');
    w = whos('-file', f);
    if any(strcmp({w.name}, 'exp_type')), fprintf('%-45s already has exp_type\n', targets{i}); continue; end
    % exp_type from any other grapes-type file of this session
    cand = [dir(fullfile(d, '*grapes*.mat')); dir(fullfile(d, 'replaced_20261008', '*grapes*.mat')); dir(fullfile(root, '*grapes*.mat')); ...
            dir(fullfile(root, 'hand selected', '*grapes*.mat')); dir(fullfile(root, 'hand selected', 'replaced_20261008', '*grapes*.mat'))];
    et = ''; src = '';
    for c = cand'
        p = fullfile(c.folder, c.name);
        if strcmp(p, f), continue; end
        ww = whos('-file', p);
        if any(strcmp({ww.name}, 'exp_type')), x = load(p, 'exp_type'); et = x.exp_type; src = p; break; end
    end
    if isempty(et), fprintf('%-45s NO exp_type source found -- left unchanged\n', targets{i}); continue; end
    G = load(f); G.exp_type = et; save(f, '-struct', 'G');
    fprintf('%-45s exp_type = %s  (from %s)\n', targets{i}, et, strrep(src, [SC '/'], ''));
end
