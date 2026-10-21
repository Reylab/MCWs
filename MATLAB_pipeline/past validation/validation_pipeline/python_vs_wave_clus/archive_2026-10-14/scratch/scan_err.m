SC = '/mnt/data0/MCWs Sessions/matlab validation/single channel';
F = {'pt 10 emu 7 ch 290/hand selected/grapes_blanks.mat', 'pt 10 emu 7 ch 362/hand selected/grapes_blanks.mat', ...
     'pt 10 emu 7 ch 365/hand selected/grapes_blanks.mat', 'pt 10 emu 7 ch 367/hand selected/grapes_blanks.mat', ...
     'pt 11 emu 17 ch 323/hand selected/grapes_blanks.mat', 'pt 11 emu 17 ch 324/hand selected/grapes_blanks.mat', ...
     'pt 11 emu 17 ch 326/hand selected/grapes_blanks.mat', 'pt 11 emu 17 ch 327/hand selected/grapes_blanks.mat', ...
     'pt 11 emu 2 ch 322/hand selected/grapes_blanks.mat', 'pt 11 emu 2 ch 323/hand selected/grapes_blanks.mat', ...
     'pt 14 emu 7 ch 364/hand selected/grapes_blanks.mat', 'pt 12 emu 18 ch 289/hand selected/Copy_of_grapes_blanks.mat', ...
     'pt 4 emu 7 ch 261 check/grapes_blanks old.mat', 'pt 6 emu 13 ch 262/grapes_blanks.mat', 'pt 6 emu 13 ch 264/grapes_blanks.mat'};
for i = 1:numel(F)
    f = fullfile(SC, F{i});
    R = load(f, 'rasters');
    ch = fieldnames(R.rasters);
    tok = regexp(F{i}, 'ch (\d+)', 'tokens', 'once'); want = str2double(tok{1});
    s = {};
    for c = ch'
        dd = R.rasters.(c{1}).details; fn = fieldnames(dd); fn = fn(~cellfun(@isempty, regexp(fn, '^class\d+$', 'once')));
        v = cellfun(@(x) dd.(x), fn)';
        s{end+1} = sprintf('%s %s', c{1}, mat2str(v)); %#ok<AGROW>
    end
    if numel(s) > 6, s = [s(1:6), {sprintf('... (%d channels)', numel(s))}]; end
    fprintf('%-58s (want chan%d): %s\n', F{i}, want, strjoin(s, ' | '));
end
