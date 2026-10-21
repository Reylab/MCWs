F = {'/mnt/data0/recluster_runs/p11_EMU-017_ch328/times_2026-10-07_23-14-38/times_mLFUS08 raw_328.mat'
     '/mnt/data0/recluster_runs/p11_EMU-017_ch328/times_2026-10-07_23-15-08/times_mLFUS08 raw_328.mat'
     '/mnt/data0/recluster_runs/p11_EMU-017_ch328/second_pass/times_2026-10-07_23-16-26/times_mLFUS08 raw_328.mat'
     '/mnt/data0/MCWs Sessions/matlab validation/single channel/pt 11 emu 17 ch 328/python_recluster_20261007_rescued/times_mLFUS08 raw_328.mat'
     '/mnt/data0/python_p10EMU07CH368/times_2026-10-06_13-46-07/times_mRFUS07 raw_368.mat'
     '/mnt/data0/MCWs Sessions/matlab validation/single channel/pt 10 emu 7 ch 368/python_recluster_20261006_sd1/times_mRFUS07 raw_368.mat'};
for i = 1:numel(F)
    T = load(F{i});
    fprintf('\n== %s\n', strrep(F{i}, '/mnt/data0/', ''));
    show(T, '', size(T.cluster_class, 1));
end
function show(S, pre, n)
    for f = fieldnames(S)'
        v = S.(f{1});
        if isstruct(v) && isscalar(v) && ~strcmp(f{1}, 'par') && ~strcmp(f{1}, 'gui_status')
            show(v, [pre f{1} '.'], n);
        else
            sz = size(v);
            flag = '';
            if any(sz == n) && ~ismember(f{1}, {'spikes','cluster_class','index','features','inspk','forced','ipermut','label','rescue_mask'}), flag = '   <-- per-spike'; end
            fprintf('  %s%s  %s %s%s\n', pre, f{1}, mat2str(sz), class(v), flag);
        end
    end
    if isfield(S, 'par')
        p = fieldnames(S.par);
        fprintf('  par.*: %s\n', strjoin(p(contains(lower(p), {'mahal','rescue','second','template','reassign','match'}))', ', '));
        if isfield(S.par, 'second_pass'), fprintf('  par.second_pass = %s\n', S.par.second_pass); end
    end
end
