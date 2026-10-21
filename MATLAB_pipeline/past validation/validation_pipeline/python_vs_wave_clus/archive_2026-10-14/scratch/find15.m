R = '/mnt/data0/MCWs Sessions/matlab validation/single channel/pt 11 emu 17 ch 328';
f = [dir(fullfile(R, '*.mat')); dir(fullfile(R, '*', '*.mat'))];
for i = 1:numel(f)
    p = fullfile(f(i).folder, f(i).name);
    if ~contains(f(i).name, {'times_', '_spikes'}), continue; end
    try, w = whos('-file', p); catch, continue; end
    v = {w.name}; s = '';
    if any(strcmp(v, 'index')), X = load(p, 'index'); s = sprintf('index n=%d', numel(X.index)); end
    if any(strcmp(v, 'cluster_class'))
        X = load(p, 'cluster_class'); lab = X.cluster_class(:,1);
        s = sprintf('%s cluster_class n=%d classes %s', s, numel(lab), mat2str(arrayfun(@(c) nnz(lab==c), 0:max(lab))));
    end
    if any(strcmp(v, 'spikes')) && ~any(strcmp(v, 'index')) && ~any(strcmp(v, 'cluster_class')), X = load(p, 'spikes'); s = sprintf('spikes %s', mat2str(size(X.spikes))); end
    fprintf('%s | %s | %s\n', datestr(f(i).datenum, 'mm-dd HH:MM'), strrep(p, [R '/'], ''), s);
end
