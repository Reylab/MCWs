function [gm_u,order] = enforce_one_to_one(gt_t,arm_t,gm,tol)
% ENFORCE_ONE_TO_ONE  Standalone copy of the local function already used
% throughout test_rescue_grape_metrics.m -- pulled out for membership_sets.m.
    % VARIABLES
    %   gt_t/arm_t/gm/tol - inputs (gm = nearest_match's raw result to clean up)
    %   order   - output: fixed reference list of GT spike positions (1:numel(gt_t))
    %   dist    - match distance (ms) for each GT spike's tentative match, NaN if unmatched
    %   ok      - boolean: which GT spikes had any tentative match
    %   seq     - GT spike indices sorted by match distance, tightest first
    %   gm_u    - output: one-to-one-cleaned match array
    %   claimed - boolean per arm spike: already given to a GT spike this pass?
    %   z       - loop variable: current GT spike index, in seq order
    order = (1:numel(gt_t))';
    dist = nan(size(gm));
    ok = gm>0;
    dist(ok) = abs(gt_t(ok)-arm_t(gm(ok)));
    [~,seq] = sort(dist);
    gm_u = zeros(size(gm));
    claimed = false(numel(arm_t),1);

    for z = seq(:)'
        if gm(z) == 0
            continue
        end
        if ~claimed(gm(z))
            gm_u(z) = gm(z);
            claimed(gm(z)) = true;
        end
    end
end
