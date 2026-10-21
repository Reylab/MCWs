addpath('/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline/validation_pipeline');
for c = {'pt12_emu18_ch299', 'pt11_emu17_ch328'}
    cfg = feval(['validation_cfg_' c{1}]);
    st = load_validation_state(cfg.out_dir);
    G = load(fullfile(cfg.gt.dir, cfg.gt.grapes_file), 'rasters');
    stimc = G.rasters.(sprintf('chan%d', cfg.ch)).(sprintf('class%d', cfg.gt.class)).stim;
    fprintf('\n== %s  GT\n', c{1});
    for s = cfg.stim_list(1:5)
        tr = stimc{s}; if iscell(tr), nt = numel(tr); else, nt = size(tr, 1); end
        no = NaN; if isKey(st.onset_map, s), no = numel(st.onset_map(s)); end
        fprintf('  stim %d: raster trials %d, onset_map trials %g\n', s, nt, no);
    end
    g = st.ref_tables{1};
    fprintf('  GT good_lat %d of %d stims; onset NaN %d; dura median %.0f\n', nnz(g.good_lat), height(g), nnz(isnan(g.onset)), median(g.dura, 'omitnan'));
    for k = 2:numel(st.ref_tables)
        a = st.ref_tables{k};
        fprintf('  arm%d good_lat %d of %d\n', k-1, nnz(a.good_lat), height(a));
    end
end
