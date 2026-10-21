R = '/mnt/data0/MCWs Sessions/matlab validation/single channel/pt 11 emu 17 ch 328';
P = '/mnt/data0/python_p11EMU17CH';
nm = 'times_mLFUS08 raw_328.mat';
arms = {'python_rescue_test','python_rescue_test_sd2','python results sd2'};
pyd = dir(fullfile(P,'times_*')); pyd = pyd([pyd.isdir]);
C = struct('name',{},'lab',{},'n',{},'resc',{},'sd',{});
for i = 1:numel(pyd)
    t = load(fullfile(P,pyd(i).name,nm),'cluster_class','quarantine','par');
    r = 0; if isfield(t,'quarantine') && isfield(t.quarantine,'rescued'), r = nnz(t.quarantine.rescued); end
    sd = NaN; if isfield(t.par,'template_sdnum'), sd = t.par.template_sdnum; end
    C(end+1) = struct('name',pyd(i).name,'lab',t.cluster_class(:,1),'n',size(t.cluster_class,1),'resc',r,'sd',sd);
    fprintf('PY %s: n=%d rescued=%d sdnum=%g classes=%d\n', pyd(i).name, C(end).n, r, sd, max(t.cluster_class(:,1)));
end
for a = 1:numel(arms)
    f = fullfile(R, arms{a}, nm);
    if ~exist(f,'file'), fprintf('ARM %s: no times file\n', arms{a}); continue; end
    t = load(f,'cluster_class','quarantine','par');
    resc = logical(t.quarantine.rescued(:)); pre = t.cluster_class(:,1) .* ~resc;
    sd = NaN; if isfield(t.par,'template_sdnum'), sd = t.par.template_sdnum; end
    sv = ''; if isfield(t.par,'save_dir'), sv = t.par.save_dir; end
    fprintf('ARM %s: n=%d rescued=%d sdnum=%g save_dir=%s\n', arms{a}, numel(pre), nnz(resc), sd, sv);
    for i = 1:numel(C)
        if C(i).n ~= numel(pre) || C(i).resc > 0, continue; end
        fprintf('    pre-rescue labels == %s ? %d\n', C(i).name, isequal(C(i).lab, pre));
    end
end
