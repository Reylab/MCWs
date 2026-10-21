SC = '/mnt/data0/MCWs Sessions/matlab validation/single channel/';
F = {'pt 4 emu 7 ch 259',259; 'pt 6 emu 13 ch 333',333; 'pt 11 emu 17 ch 328',328; 'pt 12 emu 18 ch 299',299; 'pt 12 emu 18 ch 300',300; 'pt 12 emu 18 ch 305',305};
for i = 1:size(F,1)
    d = fullfile(SC, F{i,1}, 'python_recluster_20261012_second_pass');
    t = dir(fullfile(d, 'times_*.mat')); T = load(fullfile(d, t(1).name), 'cluster_class');
    lab = T.cluster_class(:,1); tc = arrayfun(@(c) nnz(lab==c), 1:max(lab));
    files = dir(fullfile(d, 'grapes*.mat'));
    for g = files'
        w = whos('-file', fullfile(d, g.name)); G = load(fullfile(d, g.name), 'rasters');
        dt = G.rasters.(sprintf('chan%d', F{i,2})).details;
        gc = arrayfun(@(c) dt.(sprintf('class%d', c)), 1:numel(tc));
        fprintf('%-22s %-26s times %-28s grapes %-28s %s exp_type:%d\n', F{i,1}, g.name, mat2str(tc), mat2str(gc), ...
            ternary(isequal(tc,gc)), any(strcmp({w.name}, 'exp_type')));
    end
end
function s = ternary(ok), if ok, s = 'MATCH'; else, s = 'DIFF'; end, end
