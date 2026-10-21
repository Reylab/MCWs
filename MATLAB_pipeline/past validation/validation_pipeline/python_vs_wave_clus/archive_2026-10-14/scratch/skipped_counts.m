function skipped_counts()
SC = '/mnt/data0/MCWs Sessions/matlab validation/single channel';
C = {'pt 6 emu 7 ch 334', 334; 'pt 9 emu 3 ch 337', 337; 'pt 10 emu 29 ch 291', 291; 'pt 11 emu 17 ch 321', 321; 'pt 14 emu 7 ch 363', 363};
for i = 1:size(C, 1)
    d = fullfile(SC, C{i, 1}, 'hand selected');
    fprintf('\n== %s/hand selected\n', C{i, 1});
    for f = dir(fullfile(d, '*.mat'))'
        p = fullfile(f.folder, f.name);
        w = whos('-file', p);
        if any(strcmp({w.name}, 'cluster_class'))
            cc = load(p, 'cluster_class'); cc = cc.cluster_class(:, 1);
            u = unique(cc(cc > 0))';
            fprintf('  times  %-38s %s  [%s]  modified %s\n', f.name, mat2str(arrayfun(@(c) nnz(cc == c), u)), num2str(u), f.date);
        elseif any(strcmp({w.name}, 'rasters'))
            G = load(p, 'rasters'); cf = sprintf('chan%d', C{i, 2});
            if isfield(G.rasters, cf)
                dt = G.rasters.(cf).details; fn = fieldnames(dt); fn = fn(startsWith(fn, 'class') & ~contains(fn, '_'));
                n = cellfun(@(k) dt.(k), fn)'; ids = cellfun(@(k) str2double(k(6:end)), fn)';
                [ids, o] = sort(ids); n = n(o);
                fprintf('  grapes %-38s %s  [%s]  modified %s\n', f.name, mat2str(n), num2str(ids), f.date);
            else
                fprintf('  grapes %-38s no %s\n', f.name, cf);
            end
        else
            fprintf('  other  %s\n', f.name);
        end
    end
end
end
