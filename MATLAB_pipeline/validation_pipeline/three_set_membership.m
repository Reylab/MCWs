function S = three_set_membership(gt_t, arm_label, arm_t, q_mask, arm_in_scope, tol)
% THREE_SET_MEMBERSHIP  GT / arm / quarantined membership of every spike,
% for a three-circle venn.
%
%   gt_t         - GT spike times (ms) in scope (e.g. GT class 3, or every GT class > 0)
%   arm_label    - arm label per spike, full length of the arm times file (cluster_class(:,1))
%   arm_t        - arm spike times, same length (cluster_class(:,2))
%   q_mask       - logical, same length: quarantined by any rejector (quarantine.all).
%                  Not scope-dependent -- the quarantined set is the same for every venn.
%   arm_in_scope - logical, same length: spike is in the arm set for this venn
%                  (label == matched class, or label > 0 for the full set)
%   tol          - GT-to-arm spike-time match tolerance (ms)
%
% GT vs arm is the comparison; quarantine only splits its sets further:
%   1. GT is matched (by time, one-to-one) against the ARM set alone, so
%      matched / GT not in arm / arm not in GT are exactly the two-set
%      comparison (same as membership_sets / match_arms_to_ground_truth).
%   2. Arm spikes split by the quarantine flag (same file, exact): a
%      quarantined arm spike is a rescued one.
%   3. GT spikes NOT in the arm are matched one-to-one against the
%      quarantined spikes the arm did not keep, splitting GT-not-in-arm into
%      "lost" (in quarantine) and the rest.
%
% S.counts - the 7 disjoint region counts:
%   gt_only (GT not in arm, not in quarantine), arm_only (arm not in GT, not
%   rescued), q_only (quarantined, in neither), gt_arm (matched, not rescued),
%   gt_q (GT not in arm, quarantined: lost), arm_q (rescued, not in GT),
%   gt_arm_q (matched and rescued)
%   so GT not in arm = gt_only + gt_q, arm not in GT = arm_only + arm_q,
%   matched = gt_arm + gt_arm_q
% S.n_gt / S.n_arm / S.n_q - set totals
% S.q_rescued_elsewhere - quarantined spikes outside this arm set that were
%   still rescued into some other class (only nonzero for a single-cluster venn)
% S.rows.<region> - arm-file row indices of every arm-side region (all but gt_only)
% S.gt_only_pos / S.gt_lost_pos - positions in the INPUT gt_t of the GT-not-in-arm
%   spikes outside / inside quarantine

    [gt_t, gt_ord] = sort(gt_t(:));
    arm_label = arm_label(:); arm_t = arm_t(:);
    q_mask = logical(q_mask(:)); arm_in_scope = logical(arm_in_scope(:));

    % 1. GT vs arm
    a_idx = find(arm_in_scope);
    gm = nearest_match(gt_t, arm_t(a_idx), tol);
    [gm_u, ~] = enforce_one_to_one(gt_t, arm_t(a_idx), gm, tol);
    a_in_gt = false(numel(a_idx), 1);
    a_in_gt(gm_u(gm_u > 0)) = true;
    gt_rest = find(gm_u == 0);                 % GT not in arm (positions in sorted gt_t)

    % 2. arm split by quarantine (rescued)
    a_q = q_mask(a_idx);

    % 3. GT not in arm vs quarantined spikes the arm did not keep
    q_idx = find(q_mask & ~arm_in_scope);
    gm2 = nearest_match(gt_t(gt_rest), arm_t(q_idx), tol);
    [gm2_u, ~] = enforce_one_to_one(gt_t(gt_rest), arm_t(q_idx), gm2, tol);
    q_in_gt = false(numel(q_idx), 1);
    q_in_gt(gm2_u(gm2_u > 0)) = true;
    gt_lost = gm2_u > 0;

    c.gt_arm   = nnz( a_in_gt & ~a_q);
    c.gt_arm_q = nnz( a_in_gt &  a_q);
    c.arm_only = nnz(~a_in_gt & ~a_q);
    c.arm_q    = nnz(~a_in_gt &  a_q);
    c.gt_q     = nnz(gt_lost);
    c.gt_only  = numel(gt_rest) - c.gt_q;
    c.q_only   = numel(q_idx) - nnz(q_in_gt);

    S.counts = c;
    S.n_gt  = numel(gt_t);
    S.n_arm = nnz(arm_in_scope);
    S.n_q   = nnz(q_mask);
    S.q_rescued_elsewhere = nnz(q_mask & ~arm_in_scope & arm_label > 0);

    S.rows.gt_arm   = a_idx( a_in_gt & ~a_q);
    S.rows.gt_arm_q = a_idx( a_in_gt &  a_q);
    S.rows.arm_only = a_idx(~a_in_gt & ~a_q);
    S.rows.arm_q    = a_idx(~a_in_gt &  a_q);
    S.rows.gt_q     = q_idx(q_in_gt);
    S.rows.q_only   = q_idx(~q_in_gt);
    S.gt_only_pos   = gt_ord(gt_rest(~gt_lost));
    S.gt_lost_pos   = gt_ord(gt_rest(gt_lost));
end
