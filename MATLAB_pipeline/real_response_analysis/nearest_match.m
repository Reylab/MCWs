function gm = nearest_match(gt_t,ar_t,tol)
% NEAREST_MATCH  For each time in gt_t, index of its nearest time in ar_t
% within tol (0 = none). Standalone copy of the local function already used
% throughout test_rescue_grape_metrics.m -- pulled out so other new files
% (classify_three_way.m) can call it without duplicating it again.
    % VARIABLES
    %   gt_t/ar_t/tol - inputs: times to match, times to search, tolerance (ms)
    %   gm      - output: index into ar_t of each gt_t entry's nearest match (0=none within tol)
    %   as      - ar_t sorted ascending
    %   order   - permutation mapping sorted position back to original ar_t index
    %   i       - loop variable: which gt_t entry is being matched
    %   k       - search pointer into the sorted array as
    %   best/bestd - running best candidate index / its distance so far
    %   dd      - distance being tested against bestd for the current candidate
    gm = zeros(numel(gt_t),1);
    if isempty(ar_t)
        return
    end
    [as, order] = sort(ar_t(:));
    for i = 1:numel(gt_t)
        k = find(as >= gt_t(i)-tol,1,'first');
        best = 0;
        bestd = tol + 1;
        while ~isempty(k) && k <= numel(as) && as(k) <= gt_t(i) + tol
            dd = abs(as(k) - gt_t(i));
            if dd < bestd
                bestd = dd;
                best = k;
            end
            k = k+1;
        end
        if best > 0
            gm(i) = order(best);
        end
    end
end
