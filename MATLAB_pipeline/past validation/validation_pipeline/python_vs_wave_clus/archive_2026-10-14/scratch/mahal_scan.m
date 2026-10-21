f = dir('/mnt/data0/recluster_runs/**/times_*.mat');
for i = 1:numel(f)
    p = fullfile(f(i).folder, f(i).name);
    try, w = whos('-file', p); catch, continue; end
    has = any(strcmp({w.name}, 'mahal_reassigned'));
    n = NaN;
    if has, x = load(p, 'mahal_reassigned'); n = nnz(x.mahal_reassigned); end
    fprintf('%s  %-3s  %s  %s\n', datestr(f(i).datenum, 'mm-dd HH:MM'), ternary(has, 'YES', 'no'), num2str(n), strrep(p, '/mnt/data0/recluster_runs/', ''));
end
function o = ternary(c, a, b), if c, o = a; else, o = b; end, end
