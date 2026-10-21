R = dir('/mnt/data0/recluster_runs/p*');
for r = R'
    base = fullfile(r.folder, r.name);
    fs = [dir(fullfile(base, 'times_*', 'times_*.mat')); dir(fullfile(base, 'second_pass', 'times_*', 'times_*.mat'))];
    fprintf('\n== %s\n', r.name);
    for f = fs'
        T = load(fullfile(f.folder, f.name), 'cluster_class', 'quarantine', 'par');
        [~, sub] = fileparts(f.folder); if contains(f.folder, 'second_pass'), sub = ['second_pass/' sub]; end
        nres = NaN; if isfield(T, 'quarantine') && isfield(T.quarantine, 'rescued'), nres = nnz(T.quarantine.rescued); end
        sp = '-'; if isfield(T.par, 'second_pass'), sp = T.par.second_pass; if ~ischar(sp), sp = 'non-char'; end, end
        lab = T.cluster_class(:,1);
        fprintf('  %-42s %-24s n=%6d classes=%s rescued=%g\n      second_pass: %s\n', sub, f.name, numel(lab), ...
            mat2str(arrayfun(@(c) nnz(lab==c), 1:max(lab))), nres, sp);
    end
end
