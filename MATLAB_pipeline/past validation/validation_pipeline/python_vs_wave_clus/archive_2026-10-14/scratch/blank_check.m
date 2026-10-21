function blank_check()
SC = '/mnt/data0/MCWs Sessions/matlab validation/single channel';
for d = {'pt 10 emu 29 ch 291', 291; 'pt 14 emu 7 ch 363', 363; 'pt 6 emu 7 ch 334', 334}'
    G = load(fullfile(SC, d{1}, 'hand selected', 'grapes_blanks.mat'), 'ImageNames', 'rasters', 'exp_type');
    n = G.ImageNames; if iscell(n), nm = n; elseif istable(n), nm = cellstr(string(n{:, 1})); else, nm = cellstr(string(n)); end
    isb = contains(lower(nm), 'blank');
    X = G.rasters.(sprintf('chan%d', d{2})); f = fieldnames(X);
    fprintf('%-22s exp_type %s | %d image names, %d with "blank" | raster fields: %s\n', d{1}, G.exp_type, numel(nm), nnz(isb), strjoin(f', ' '));
end
end
