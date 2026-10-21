SC = '/mnt/data0/MCWs Sessions/matlab validation/single channel/';
G = {'pt 4 emu 7 ch 259', 259, 1, 55275, 'hand selected'
     'pt 6 emu 13 ch 333', 333, 1, 8464, 'hand selected'
     'pt 9 emu 3 ch 331 check', 331, 4, 3538, 'hand selected'
     'pt 9 emu 3 ch 331 check', 331, 4, 3538, 'hand selected rebuilt 4class'
     'zneed rerun grapes blanks changednew trial pt 9 emu 3 ch 335', 335, 2, 8159, 'hand selected'
     'pt 10 emu 7 ch 368', 368, 3, 10341, 'hand selected'
     'pt 11 emu 17 ch 328', 328, 1, 4066, 'hand selected'
     'pt 12 emu 18 ch 299', 299, 1, 4161, 'hand selected'
     'pt 12 emu 18 ch 300', 300, 1, 9013, 'hand selected'
     'pt 12 emu 18 ch 305', 305, 1, 1357, 'hand selected'};
for i = 1:size(G, 1)
    d = fullfile(SC, G{i,1}, G{i,5});
    t = dir(fullfile(d, 'times_*.mat'));
    T = load(fullfile(d, t(1).name), 'cluster_class');
    nt = arrayfun(@(c) nnz(T.cluster_class(:,1) == c), 1:max(T.cluster_class(:,1)));
    gf = fullfile(d, 'grapes_blanks.mat');
    R = load(gf, 'rasters'); dd = R.rasters.(sprintf('chan%d', G{i,2})).details;
    f = fieldnames(dd); f = f(~cellfun(@isempty, regexp(f, '^class\d+$', 'once')));
    ng = zeros(1, numel(f));
    for k = 1:numel(f), ng(str2double(f{k}(6:end))) = dd.(f{k}); end
    c = G{i,3}; want = G{i,4};
    vt = NaN; if c <= numel(nt), vt = nt(c); end
    vg = NaN; if c <= numel(ng), vg = ng(c); end
    fprintf('\n%s / %s  -- you: class %d = %d\n', G{i,1}, G{i,5}, c, want);
    fprintf('  times  (%s) class %d = %s %s | all: %s\n', t(1).name, c, num2str(vt), ok(vt == want), mat2str(nt));
    fprintf('  grapes (grapes_blanks)        class %d = %s %s | all: %s\n', c, num2str(vg), ok(vg == want), mat2str(ng));
    if vt ~= want, fprintf('  times classes equal to %d: %s\n', want, mat2str(find(nt == want))); end
    if vg ~= want, fprintf('  grapes classes equal to %d: %s\n', want, mat2str(find(ng == want))); end
end
function s = ok(b), if b, s = 'OK'; else, s = 'DIFFERENT'; end, end
