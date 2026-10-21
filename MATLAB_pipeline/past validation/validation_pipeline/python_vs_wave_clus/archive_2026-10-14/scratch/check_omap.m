addpath('/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline/real_response_analysis', '/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline/validation_pipeline');
for c = {'pt11_emu17_ch328', 'pt6_emu13_ch333', 'pt4_emu7_ch259', 'pt9_emu3_ch331', 'pt12_emu18_ch299'}
    cfg = feval(['validation_cfg_' c{1}]);
    S = load(fullfile(cfg.root, 'stimulus.mat')); F = load(fullfile(cfg.root, 'finalevents.mat'));
    E = load(fullfile(cfg.root, 'experiment_properties_online3.mat'));
    N = min(numel(S.stimulus), numel(E.scr_end_cell));
    a = build_all_stim_onset_map(S, F, N, cfg.stim_list);
    b = build_all_stim_onset_map(S, F, N, cfg.stim_list, E);
    same = a.Count == b.Count && all(cellfun(@(k) isequal(a(k), b(k)), a.keys));
    G = load(fullfile(cfg.gt.dir, cfg.gt.grapes_file), 'rasters');
    st = G.rasters.(sprintf('chan%d', cfg.ch)).(sprintf('class%d', cfg.gt.class)).stim;
    ok = 0;
    for s = cfg.stim_list
        x = st{s}; if iscell(x), nt = numel(x); else, nt = size(x, 1); end
        ok = ok + (isKey(b, s) && numel(b(s)) == nt);
    end
    fprintf('%s: old keys %d, new keys %d of %d, old==new %d, trial counts match raster %d of %d\n', ...
        c{1}, a.Count, b.Count, numel(cfg.stim_list), same, ok, numel(cfg.stim_list));
end
