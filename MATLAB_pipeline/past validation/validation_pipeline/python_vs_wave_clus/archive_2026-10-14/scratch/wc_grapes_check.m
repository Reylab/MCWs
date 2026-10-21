function wc_grapes_check()
SC = '/mnt/data0/MCWs Sessions/matlab validation/single channel';
for d = {'pt 6 emu 7 ch 334', 'wave_clus1', 334, 'times_mLFUS05 raw_334.mat'; 'pt 6 emu 7 ch 334', 'wave_clus5', 334, 'times_mLFUS05 raw_334.mat'; 'pt 9 emu 3 ch 337', 'wave_clus1', 337, 'times_mLFUS08 raw_337.mat'}'
    D = fullfile(SC, d{1}, d{2});
    cc = load(fullfile(D, d{4}), 'cluster_class'); cc = cc.cluster_class(:, 1); u = unique(cc(cc > 0))';
    G = load(fullfile(D, 'grapes_blanks.mat'), 'rasters'); dt = G.rasters.(sprintf('chan%d', d{3})).details;
    g = arrayfun(@(c) iff(isfield(dt, sprintf('class%d', c)), dt, c), u);
    fprintf('%s/%s: times %s | grapes %s | %s\n', d{1}, d{2}, mat2str(arrayfun(@(c) nnz(cc == c), u)), mat2str(g), ...
        tern(isequal(arrayfun(@(c) nnz(cc == c), u), g), 'MATCH', 'DIFF'));
end
end
function v = iff(ok, dt, c), if ok, v = dt.(sprintf('class%d', c)); else, v = NaN; end, end
function v = tern(c, a, b), if c, v = a; else, v = b; end, end
