function verify_counts()
SC = '/mnt/data0/MCWs Sessions/matlab validation/single channel';
C = {'pt 6 emu 7 ch 334', [1 2 3], [3897 1826 1017]; 'pt 9 emu 3 ch 337', [3 4], [8130 5090]; 'pt 10 emu 29 ch 291', 2, 4552; ...
     'pt 10 emu 29 ch 296', 2, 5467; 'pt 10 emu 29 ch 368', 1, 6074; 'pt 11 emu 17 ch 321', 2, 5249; ...
     'pt 14 emu 7 ch 291', 1, 3855; 'pt 14 emu 7 ch 363', 3, 2877; 'pt 14 emu 7 ch 364', 5, 2540; 'pt 14 emu 7 ch 369', 3, 1209};
for i = 1:size(C, 1)
    d = fullfile(SC, C{i, 1});
    t = dir(fullfile(d, 'hand selected', 'times_*.mat'));
    g = dir(fullfile(d, 'hand selected', 'grapes_blanks.mat'));
    nc5 = dir(fullfile(d, '*.NC5'));
    sess = all(cellfun(@(f) exist(fullfile(d, f), 'file') == 2, {'NSx.mat', 'finalevents.mat', 'experiment_properties_online3.mat'}));
    if numel(t) ~= 1, fprintf('%-22s %d times files in hand selected\n', C{i, 1}, numel(t)); continue; end
    cc = load(fullfile(t.folder, t.name), 'cluster_class'); cc = cc.cluster_class(:, 1);
    got = arrayfun(@(c) nnz(cc == c), C{i, 2});
    gcount = '';
    if ~isempty(g)
        G = load(fullfile(g.folder, g.name));
        if isfield(G, 'grapes')
            ch = regexp(C{i, 1}, 'ch (\d+)', 'tokens', 'once'); ch = str2double(ch{1});
            fn = fieldnames(G.grapes);
            k = fn(startsWith(fn, 'ch'));
            if ~isempty(k)
                X = G.grapes.(k{1});
                gc = arrayfun(@(c) iff(isfield(X, sprintf('class%d', c)), @() numel(X.(sprintf('class%d', c)).spikes_time), @() NaN), C{i, 2}, 'ErrorHandler', @(varargin) NaN);
                gcount = mat2str(gc);
            end
        end
    end
    fprintf('%-22s %-5s times %s expected %s | grapes_blanks %d %s | NC5 %d | session files %d | %s\n', C{i, 1}, ...
        iff(isequal(got, C{i, 3}), @() 'MATCH', @() 'DIFF'), mat2str(got), mat2str(C{i, 3}), ~isempty(g), gcount, numel(nc5), sess, t.name);
end
end
function v = iff(c, a, b)
if c, v = a(); else, v = b(); end
end
