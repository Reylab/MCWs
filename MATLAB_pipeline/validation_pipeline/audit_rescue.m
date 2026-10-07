function A = audit_rescue(T, cls, ref_ms)
% AUDIT_RESCUE  Replay a Python arm's rescue exactly and explain how each
% rescued spike got its class.
%
%   A = audit_rescue(T, cls, ref_ms)
%
% Replays mcws.clustering.rescue with the settings saved in T.par
% (template_sdnum, first_pass_sdnum, amp_pct_range, mahal_reassign), using
% force_membership_wc's 'center' path (a line-for-line port of
% force_membership.py) on the same inputs Python used:
%   templates  - every pre-rescue labeled spike (cluster_class, rescued ones removed)
%   candidates - quarantined spikes that were label 0 before rescue
%   Mahalanobis space - T.features (what rescue.py prefers), else T.inspk
% Then compares the replayed labels with Python's, spike by spike, and runs
% the same replay with Mahalanobis OFF to see which rescued spikes the
% Euclidean template match put in their final class and which the
% Mahalanobis step moved there.
%
%   cls    - class to audit in detail (e.g. the arm's matched class)
%   ref_ms - refractory period (ms) for the ISI check (default T.par.ref_ms or 1.5)
%
% A.n_cand / A.n_mismatch       - candidates, and how many replayed labels differ from Python
% A.how                         - table, one row per final class of the rescued spikes:
%                                 n rescued / by template match / moved in by Mahalanobis
% A.cls_* (for CLS)             - per-rescued-spike detail: Euclidean label,
%                                 distance to CLS in units of its refined std,
%                                 width/refractory flags, and ISI to the nearest
%                                 non-rescued CLS spike
    lab  = T.cluster_class(:,1);
    tms  = T.cluster_class(:,2);
    resc = logical(T.quarantine.rescued(:));
    q    = logical(T.quarantine.all(:));
    W    = double(T.spikes);
    if isfield(T, 'features') && size(T.features,1) == numel(lab)
        F = double(T.features);
    else
        F = double(T.inspk);
    end
    p = T.par;
    if nargin < 3 || isempty(ref_ms)
        ref_ms = 1.5; if isfield(p, 'ref_ms'), ref_ms = p.ref_ms; end
    end

    pre = lab .* ~resc;                 % labels before rescue
    valid = pre > 0;
    cand = q & pre == 0;
    if isfield(p, 'rescue_refractory') && ~p.rescue_refractory && isfield(T.quarantine, 'refractory_period')
        cand = cand & ~logical(T.quarantine.refractory_period(:));   % rescue skipped these
    end

    par.template_type = 'center';
    par.template_sdnum = p.template_sdnum;
    par.template_first_pass_sdnum = p.first_pass_sdnum;
    par.template_amp_pct_range = double(p.amp_pct_range(:)');
    par.template_mahal_reassign = logical(p.mahal_reassign);
    lab_rep = force_membership_wc(W(valid,:), pre(valid)', W(cand,:), par, F(valid,:), F(cand,:));
    par.template_mahal_reassign = false;
    lab_euc = force_membership_wc(W(valid,:), pre(valid)', W(cand,:), par);

    lab_py = lab(cand)';
    A.n_cand = nnz(cand);
    A.n_mismatch = nnz(lab_rep ~= lab_py);
    A.n_rescued_py = nnz(lab_py > 0);
    A.n_rescued_replay = nnz(lab_rep > 0);

    cls_list = unique(lab_py(lab_py > 0));
    n = arrayfun(@(c) nnz(lab_py == c), cls_list);
    by_tm = arrayfun(@(c) nnz(lab_py == c & lab_euc == c), cls_list);
    by_mh = arrayfun(@(c) nnz(lab_py == c & lab_euc ~= c), cls_list);
    A.how = table(cls_list(:), n(:), by_tm(:), by_mh(:), ...
        'VariableNames', {'class', 'n_rescued', 'by_template_match', 'moved_in_by_mahalanobis'});

    % detail for CLS: distance to its refined (pass-1) template in std units
    ci = find(cand);
    r_cls = ci(lab_py == cls);
    [~, ~, info] = sd1_pass_templates(T, cls, p.first_pass_sdnum, double(p.amp_pct_range(:)'));
    core_res = false(size(lab)); core_res(info.res_core_rows) = true;
    members = (pre == cls) | core_res;  % refined template = members + rescue pass-1 core
    C = mean(W(members,:), 1);
    sd = sqrt(sum(var(W(members,:), 1, 1)));
    A.cls_rows = r_cls;
    A.cls_euclid_label = lab_euc(lab_py == cls)';
    A.cls_dist_sd = sqrt(sum((W(r_cls,:) - C).^2, 2)) / sd;
    [M, names] = quarantine_property_flags(T);
    A.cls_fail_width = M(r_cls, strcmp(names, 'bad width'));
    A.cls_fail_refract = M(r_cls, strcmp(names, 'refractory'));
    kept = sort(tms(lab == cls & ~resc));
    A.cls_isi_ms = arrayfun(@(t) min(abs(kept - t)), tms(r_cls));
    A.ref_ms = ref_ms;
end
