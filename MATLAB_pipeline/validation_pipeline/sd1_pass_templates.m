function [tmpl_clu, tmpl_res, info] = sd1_pass_templates(T, cls, sdnum, amp_pct)
% SD1_PASS_TEMPLATES  Rebuild the sd1 (first-pass) template of class CLS from
% a Python-layout times file, for both places the two-pass template match runs:
%
%   tmpl_clu - clustering force-membership, pass 1
%              seeds:      SPC members (label > 0, not forced, not quarantined)
%              candidates: every other clean spike (forced or left at 0)
%   tmpl_res - rescue, pass 1
%              seeds:      every pre-rescue labeled spike (forced ones included)
%              candidates: the quarantined spikes (minus refractory-flagged
%                          ones when the run saved rescue_refractory = 0)
%
% Each is mean(seeds of CLS + candidates pass 1 assigns to CLS), i.e. the
% template pass 2 actually runs against. Pass 1 is reproduced exactly as in
% mcws/utils/template_matching.py: nearest template among ALL classes, within
% sdnum x that class's distance std, and inside the [1 99] percentile
% amplitude window at the template peak. Neither template is saved by the
% pipeline, which is why they're rebuilt here. Plot use only.
%
%   T      - loaded times file: spikes, cluster_class, forced, quarantine.all/.rescued
%   cls    - class to return
%   sdnum  - first-pass radius (default 1)
%   amp_pct- amplitude gate percentiles (default [1 99], [] = off)
%   info   - counts: n seeds / n core for each pass, plus .clu_core_rows /
%            .res_core_rows: times-file rows of each pass's sd1 core for CLS

    if nargin < 3 || isempty(sdnum), sdnum = 1; end
    if nargin < 4, amp_pct = [1 99]; end

    W = double(T.spikes);
    lab = T.cluster_class(:,1);
    q = logical(T.quarantine.all(:));
    if isfield(T.quarantine, 'rescued')
        resc = logical(T.quarantine.rescued(:));
    else
        resc = false(size(T.cluster_class, 1), 1);   % no rescue step (clustering-only run)
    end
    forced = logical(T.forced(:));

    % clustering pass 1
    spc_lab = lab .* (~forced & ~q & ~resc);
    cand_clu = ~q & spc_lab == 0;
    [tmpl_clu, info.clu_seed, info.clu_core, info.clu_core_rows] = pass1_template(W, spc_lab, cand_clu, cls, sdnum, amp_pct);

    % rescue pass 1
    pre_lab = lab .* ~resc;
    cand_res = q & pre_lab == 0;
    cand_res = cand_res & ~refractory_excluded(T);
    [tmpl_res, info.res_seed, info.res_core, info.res_core_rows] = pass1_template(W, pre_lab, cand_res, cls, sdnum, amp_pct);
end

function [tmpl, n_seed, n_core, core_rows] = pass1_template(W, lab, cand, cls, sdnum, amp_pct)
    K = max(lab);
    ns = size(W,2);
    C = zeros(K, ns); sd = zeros(K,1);
    for k = 1:K
        Wk = W(lab == k, :);
        if isempty(Wk), C(k,:) = inf; continue; end
        C(k,:) = mean(Wk, 1);
        sd(k) = sqrt(sum(var(Wk, 1, 1)));
    end
    Wc = W(cand, :);
    D = sqrt(max(sum(Wc.^2,2) + sum(C.^2,2)' - 2*Wc*C', 0));
    within = D < (sdnum*sd)';
    if ~isempty(amp_pct)
        [~, pk] = max(abs(C), [], 2);
        lo = -inf(1,K); hi = inf(1,K);
        for k = 1:K
            a = W(lab == k, pk(k));
            if numel(a) < 2, continue; end
            lo(k) = prctile_linear(a, amp_pct(1));
            hi(k) = prctile_linear(a, amp_pct(2));
        end
        A = Wc(:, pk);
        within = within & A >= lo & A <= hi;
    end
    D(~within) = inf;
    [best, idx] = min(D, [], 2);
    core = isfinite(best) & idx == cls;
    seeds = W(lab == cls, :);
    tmpl = mean([seeds; Wc(core,:)], 1);
    n_seed = size(seeds,1); n_core = nnz(core);
    cand_rows = find(cand);
    core_rows = cand_rows(core);
end

function v = prctile_linear(x, p)
% numpy-style linear percentile (matches template_matching.py)
    x = sort(x(:)); n = numel(x);
    pos = p/100*(n-1); lo = floor(pos); hi = ceil(pos);
    v = x(lo+1)*(1-(pos-lo)) + x(hi+1)*(pos-lo);
end

function x = refractory_excluded(T)
% refractory-flagged spikes the rescue skipped (runs saved with
% par.rescue_refractory = 0); all-false for older runs that rescued them
    n = size(T.cluster_class, 1);
    x = false(n, 1);
    if isfield(T, 'par') && isfield(T.par, 'rescue_refractory') && ~T.par.rescue_refractory ...
            && isfield(T.quarantine, 'refractory_period')
        x = logical(T.quarantine.refractory_period(:));
    end
end
