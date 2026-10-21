F = { ...
 '/mnt/data0/MCWs Sessions/matlab validation/single channel/pt 10 emu 7 ch 368/python_rescue_test_sd1/times_mRFUS07 raw_368.mat', ...
 '/mnt/data0/MCWs Sessions/matlab validation/single channel/pt 10 emu 7 ch 368/python_rescue_no_refract_sd1/times_mRFUS07 raw_368.mat', ...
 '/mnt/data0/MCWs Sessions/matlab validation/single channel/pt 11 emu 17 ch 328/python_rescue_test/times_mLFUS08 raw_328.mat', ...
 '/mnt/data0/MCWs Sessions/matlab validation/single channel/pt 11 emu 17 ch 328/python_rescue_no_refract_sd1/times_mLFUS08 raw_328.mat'};
for i = 1:numel(F)
    t = load(F{i}, 'par', 'quarantine_properties', 'quarantine');
    fprintf('\n== %s\n', strrep(F{i}, '/mnt/data0/MCWs Sessions/matlab validation/single channel/', ''));
    p = t.par;
    if isfield(p, 'qc_params'), q = p.qc_params; fn = fieldnames(q);
        for j = 1:numel(fn)
            if contains(lower(fn{j}), 'amp'), v = q.(fn{j}); if ischar(v), s = v; else, s = mat2str(v); end
                fprintf('  par.qc_params.%s = %s\n', fn{j}, s); end
        end
    else, fprintf('  par.qc_params: (not saved)\n'); end
    if isfield(p, 'artifact_rejection_used'), a = p.artifact_rejection_used; fn = fieldnames(a);
        for j = 1:numel(fn), v = a.(fn{j}); if ischar(v), s = strjoin(cellstr(v), ','); else, s = mat2str(v); end
            fprintf('  par.artifact_rejection_used.%s = %s\n', fn{j}, s); end
    end
    qp = t.quarantine_properties;
    lla = arrayfun(@(x) double(x.low_low_amp_spike), qp(:));
    p20 = arrayfun(@(x) double(x.peak_sample_20), qp(:));
    valid = isfinite(p20);
    fprintf('  low-amp flagged: %d of %d spikes with a valid peak = %.2f%%  (within_channel total %d)\n', ...
        nnz(lla(valid)), nnz(valid), 100*nnz(lla(valid))/nnz(valid), nnz(t.quarantine.within_channel));
end
