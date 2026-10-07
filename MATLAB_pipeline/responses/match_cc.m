function [T, DET, COMP] = match_cc(ccg, cca, ch, tol, min_recall)
% MATCH_CC  Match every ground-truth class to an arm class by one-to-one
% nearest spike-time overlap on two [class time_ms] matrices, and fully
% account for where each class's spikes go. Shared by
% validate_recluster_vs_handselect.m and analyze_cluster_membership.m.
%
% INPUTS
%   ccg, cca    [class time_ms] matrices (class column 1, time_ms column 2),
%               same shape as reconstruct_class_cc's output / cluster_class.
%   ch          channel number (carried into the output tables, not used
%               for matching itself)
%   tol         one-to-one spike-time match tolerance (ms)
%   min_recall  a ground-truth class is "matched" when the best arm class's
%               recall >= this; otherwise "merged"/"split"/"lost" per the
%               status logic below.
%
% OUTPUT
%   T    - one row per ground-truth class (the headline match + where its
%          spikes ended up + what the matched arm cluster is made of):
%          channel_id, gt_class, arm_class, n_gt, n_arm, n_shared, recall,
%          precision, arm_class_dominant_gt, status, n_gt_unmatched,
%          frac_unmatched, frac_elsewhere, n_arm_from_other_gt, n_arm_new
%   DET  - long: for each gt class, the spike count landing in every
%          destination (arm classK / arm class0 / unmatched) -- "where did
%          this GT class's spikes go"
%   COMP - long: for each gt class's matched arm cluster, how many of its
%          spikes come from this gt class / other gt classes / gt class0 /
%          are new (no gt counterpart at all) -- "what is the matched arm
%          cluster made of"

    gt_t = ccg(:,2); gt_k = ccg(:,1);
    ar_t = cca(:,2); ar_k = cca(:,1);
    gt_cls = sort(unique(gt_k(gt_k > 0)))';
    ar_cls = sort(unique(ar_k(ar_k > 0)))';
    assert(numel(gt_cls) <= 100 && numel(ar_cls) <= 100, ...
        ['match_cc: %d/%d distinct class labels - the cluster_class column order ' ...
         'is wrong (expected [class time_ms]).'], numel(gt_cls), numel(ar_cls));
    ngt = numel(gt_cls);

    [gt_dest, ar_src] = assign_pairs(gt_t, ar_t, ar_k, gt_k, tol);
    % gt_dest(p) = arm class number matched to GT spike p (NaN = unmatched)
    % ar_src(q)  = GT  class number matched to arm spike q (NaN = "new" spike)

    T = mt_schema();
    DET = table('Size', [0 5], 'VariableTypes', {'double','double','string','double','double'}, ...
                'VariableNames', {'channel_id','gt_class','dest','n_spikes','frac_of_gt_class'});
    COMP = table('Size', [0 6], 'VariableTypes', ...
                 {'double','double','double','string','double','double'}, ...
                 'VariableNames', {'channel_id','gt_class','arm_class','src','n_spikes','frac_of_arm_cluster'});
    if ngt == 0, return; end

    for i = 1:ngt
        c = gt_cls(i);
        sel = gt_k == c;  n_gt = nnz(sel);
        dest = gt_dest(sel);
        cnt = zeros(1, numel(ar_cls));
        for jj = 1:numel(ar_cls), cnt(jj) = nnz(dest == ar_cls(jj)); end
        n_unmatched = nnz(isnan(dest));
        n_to_c0     = nnz(dest == 0);

        if isempty(cnt) || max(cnt) == 0
            row = {ch, c, 0, n_gt, 0, 0, 0, 0, 0, "lost", n_unmatched, ...
                   n_unmatched / max(n_gt,1), 0, 0, 0};
            T = [T; row];
            DET = [DET; {ch, c, "unmatched", n_unmatched, n_unmatched/max(n_gt,1)}];
            continue;
        end

        [best_n, bj] = max(cnt);
        arm_class = ar_cls(bj);
        recall    = best_n / max(n_gt,1);
        n_elsewhere = sum(cnt) - best_n;

        selq = ar_k == arm_class;  n_arm = nnz(selq);
        src  = ar_src(selq);
        n_from_c        = nnz(src == c);
        n_from_other_gt = nnz(src > 0 & src ~= c);
        n_from_gt0      = nnz(src == 0);
        n_new           = nnz(isnan(src));
        precision = n_from_c / max(n_arm,1);
        if any(src > 0)
            dominant_gt = mode(src(src > 0));
        else
            dominant_gt = 0;
        end

        rj = cnt / max(n_gt,1);
        if recall >= min_recall
            if precision >= 0.5, status = "matched"; else, status = "merged"; end
        elseif nnz(rj >= 0.2) >= 2
            status = "split";
        else
            status = "lost";
        end

        T = [T; {ch, c, arm_class, n_gt, n_arm, n_from_c, recall, precision, ...
                 dominant_gt, status, n_unmatched, n_unmatched/max(n_gt,1), ...
                 n_elsewhere/max(n_gt,1), n_from_other_gt, n_new}];

        for jj = 1:numel(ar_cls)
            if cnt(jj) == 0, continue; end
            DET = [DET; {ch, c, "class" + string(ar_cls(jj)), cnt(jj), cnt(jj)/max(n_gt,1)}];
        end
        if n_to_c0 > 0
            DET = [DET; {ch, c, "class0", n_to_c0, n_to_c0/max(n_gt,1)}];
        end
        if n_unmatched > 0
            DET = [DET; {ch, c, "unmatched", n_unmatched, n_unmatched/max(n_gt,1)}];
        end

        f = @(n) n / max(n_arm,1);
        COMP = [COMP; ...
            {ch, c, arm_class, "gt_class" + string(c),      n_from_c,        f(n_from_c)}; ...
            {ch, c, arm_class, "gt_other",                  n_from_other_gt, f(n_from_other_gt)}; ...
            {ch, c, arm_class, "gt_class0",                 n_from_gt0,      f(n_from_gt0)}; ...
            {ch, c, arm_class, "new_no_gt_spike",           n_new,           f(n_new)}];
    end
end

function T = mt_schema()
    T = table('Size', [0 15], 'VariableTypes', ...
        {'double','double','double','double','double','double','double','double', ...
         'double','string','double','double','double','double','double'}, ...
        'VariableNames', {'channel_id','gt_class','arm_class','n_gt','n_arm','n_shared', ...
         'recall','precision','arm_class_dominant_gt','status','n_gt_unmatched', ...
         'frac_unmatched','frac_elsewhere','n_arm_from_other_gt','n_arm_new'});
end

function [gt_dest, ar_src] = assign_pairs(gt_t, ar_t, ar_k, gt_k, tol)
% One-to-one nearest match (<= tol ms) over ALL spikes of both files
% (any class, incl 0). Each arm spike is used at most once.
    [gs, go] = sort(gt_t(:));
    [as, ao] = sort(ar_t(:));
    na = numel(as);
    consumed = false(na, 1);
    gt_dest = nan(numel(gt_t), 1);
    ar_src  = nan(numel(ar_t), 1);
    j = 1;
    for i = 1:numel(gs)
        lo = gs(i) - tol; hi = gs(i) + tol;
        while j <= na && as(j) < lo, j = j + 1; end
        k = j; best = -1; bestd = tol + 1;
        while k <= na && as(k) <= hi
            if ~consumed(k)
                dd = abs(as(k) - gs(i));
                if dd < bestd, bestd = dd; best = k; end
            end
            k = k + 1;
        end
        if best > 0
            consumed(best) = true;
            gt_dest(go(i)) = ar_k(ao(best));
            ar_src(ao(best)) = gt_k(go(i));
        end
    end
end
