% Reduce the grapes files copied into hand selected (2026-10-08) to the
% folder's own channel: rasters.chan<ch> only, other top-level fields kept.
SC = '/mnt/data0/MCWs Sessions/matlab validation/single channel';
J = {'need_times/pt 9 emu 3 ch 332', 332, 'grapes - Copy.mat'
     'mismatch/pt 10 emu 7 ch 296', 296, 'Copy_of_grapes.mat'};
for i = 1:size(J, 1)
    [fold, ch, name] = J{i, :};
    f = fullfile(SC, fold, 'hand selected', name);
    try
        G = load(f);
        cf = sprintf('chan%d', ch);
        before = fieldnames(G.rasters);
        d = G.rasters.(cf).details;
        fn = fieldnames(d); fn = fn(~cellfun(@isempty, regexp(fn, '^class\d+$', 'once')));
        c0 = cellfun(@(x) d.(x), fn)';
        G.rasters = struct(cf, G.rasters.(cf));
        save(f, '-struct', 'G', '-v7.3');
        H = load(f, 'rasters'); d2 = H.rasters.(cf).details;
        c1 = cellfun(@(x) d2.(x), fn)';
        b = dir(f);
        assert(isequal(c0, c1) && numel(fieldnames(H.rasters)) == 1, 'check failed');
        fid = fopen(fullfile(SC, fold, 'hand selected', 'README_files_20261008.txt'), 'a');
        fprintf(fid, 'trimmed: %s to %s only (was %d channels); class counts %s\n', name, cf, numel(before), mat2str(c1));
        fclose(fid);
        fprintf('[%s] %s: %d channels -> %s only, %.1f MB, counts %s OK\n', fold, name, numel(before), cf, b.bytes/1e6, mat2str(c1));
    catch ME
        fprintf('[%s] %s FAILED: %s\n', fold, name, ME.message);
    end
end
