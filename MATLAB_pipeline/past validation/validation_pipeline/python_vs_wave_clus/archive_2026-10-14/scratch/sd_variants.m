addpath('/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline/batch_files/clustering');
T = load('/mnt/data0/python_p10EMU07CH368/times_2026-10-06_13-46-07/times_mRFUS07 raw_368.mat');
lab = T.cluster_class(:,1); resc = logical(T.quarantine.rescued(:)); q = logical(T.quarantine.all(:));
forced = logical(T.forced(:)); W = double(T.spikes); F = double(T.features);
pre = lab .* ~resc; cand = q & pre == 0; py = lab(cand)';
seeds = struct('name', {'all pre-rescue labeled', 'SPC originals only (not forced)'}, ...
               'mask', {pre > 0, pre > 0 & ~forced});
best = '';
for s = seeds
    for sd = [1 2]
        for mh = [true false]
            for amp = {[1 99], []}
                par = struct('template_type','center','template_sdnum',sd,'template_first_pass_sdnum',1, ...
                             'template_amp_pct_range',amp{1},'template_mahal_reassign',mh);
                v = s.mask;
                if mh
                    out = force_membership_wc(W(v,:), pre(v)', W(cand,:), par, F(v,:), F(cand,:));
                else
                    out = force_membership_wc(W(v,:), pre(v)', W(cand,:), par);
                end
                mm = nnz(out ~= py);
                fprintf('%-32s sd%d mahal=%d amp=%-6s rescued %5d  mismatches %5d\n', s.name, sd, mh, mat2str(amp{1}), nnz(out>0), mm);
            end
        end
    end
end
fprintf('python rescued %d\n', nnz(py > 0));
