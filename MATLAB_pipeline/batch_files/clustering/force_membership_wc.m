function class_out = force_membership_wc(f_in, class_in, f_out, par)
% class = function force_membership_wc(f_in, class_in, f_out, par)
% Given classified points, try to classify new points via template matching
%
% f_in:          features of classified points  (# input spikes x n_features)
% class_in:      classification of those points
% f_out:         features of points to be classified (nspk x n_features)
% par        environment variables, of which the following are
%                required:
%                    o par.template_sdnum - max radius of cluster,
%                                                   in std devs.
%                    o par.template_k     - # of nearest neighbors
%                    o par.template_k_min - min # of nn for vote
%                    o par.template_type  - nn, center, ml, mahal
%
% For par.template_type == 'center' the following optional fields select the
% two-pass pipeline ported from force_membership.py (defaults applied when the
% field is missing/empty):
%                    o par.template_first_pass_sdnum  (default 1)
%                    o par.template_amp_pct_range     (default [1 99], [] = off)
%                    o par.template_mahal_reassign    (default true)
%
% 'center' two-pass logic:
%   1. build initial templates + distance std from f_in / class_in.
%   2. first pass: assign only f_out within first_pass_sdnum*std (tight core).
%   3. re-estimate templates + std from f_in members PLUS the first-pass core.
%   4. second pass: assign all f_out within template_sdnum*(refined std).
%   5. (optional) keep the euclidean accept/reject, then move every accepted
%      spike to the cluster it is closest to in waveform-space (shrunk)
%      Mahalanobis distance.

nspk = size(f_out,1);
class_out = zeros(1,size(f_out,1));
switch par.template_type
    case 'nn'
        sdnum = par.template_sdnum;
        k     = par.template_k;
        k_min = par.template_k_min;
        sd    = sqrt(sum(var(f_in,1)))*ones(1,size(f_in,1));
        for i=1:nspk,
            nn = nearest_neighbor(f_out(i,:),f_in,sdnum*sd,Inf*ones(size(f_in)),Inf,k);
            if( nn )
                winner = mode(class_in(nn));
                if nnz(class_in(nn)==winner)<k_min
                    class_out(i) = 0;
                else
                    class_out(i) = winner;
                end
            else
                class_out(i) = 0;
            end
        end

    case 'center'
        sdnum    = par.template_sdnum;
        fp_sdnum = local_getdef(par, 'template_first_pass_sdnum', 1);
        amp_pct  = local_getdef(par, 'template_amp_pct_range', [1 99]);
        do_mahal = local_getdef(par, 'template_mahal_reassign', true);

        class_in = class_in(:)';                       % normalize to row

        % --- initial templates from the classified spikes ---
        [centers, sd, ~] = build_templates(class_in, f_in);
        if isempty(centers) || nspk == 0
            return                                     % class_out already zeros
        end

        % --- pass 1: tight assignment -> confident core ---
        labels_p1 = assign_to_templates(f_out, centers, fp_sdnum*sd, ...
                                        f_in, class_in, amp_pct);

        % --- re-estimate templates / std from members + core ---
        core = labels_p1 > 0;
        if any(core)
            comb_wf  = [f_in;  f_out(core,:)];
            comb_cls = [class_in, labels_p1(core)];
        else
            comb_wf  = f_in;
            comb_cls = class_in;
        end
        [centers2, sd2, ~] = build_templates(comb_cls, comb_wf);
        if isempty(centers2)
            centers2 = centers;  sd2 = sd;
        end

        % --- pass 2: final assignment at template_sdnum x refined std ---
        labels = assign_to_templates(f_out, centers2, sdnum*sd2, ...
                                     comb_wf, comb_cls, amp_pct);

        % --- optional post-step: waveform-space Mahalanobis reassignment ---
        if do_mahal
            if any(core)
                comb_feat = [f_in; f_out(core,:)];
            else
                comb_feat = f_in;
            end
            labels = mahalanobis_reassign(comb_cls, comb_feat, f_out, labels);
        end

        class_out = labels(:)';

    case 'ml'
        [mu inv_sigma] = fit_gaussian(f_in,class_in);
        for i=1:nspk,
            class_out(i) = ML_gaussian(f_out(i,:),mu,inv_sigma);
        end
    case 'mahal'
        classes = unique(class_in);
        mdistance = zeros(length(classes), nspk);
        maxdist   = zeros(1, length(classes));
        for ci = 1:length(classes)
           i = classes(ci);
           mdistance(i,:) = mahal(f_out, f_in(class_in ==i, :));
           maxdist(i) = sqrt(mean(mahal(f_in(class_in ==i, :), f_in(class_in ==i, :))));
        end
        sdnum = par.template_sdnum;
        for i = 1:nspk
             [d winner] = min(mdistance(:,i));
             if sqrt(d) < sdnum*maxdist(winner)
                 class_out(i) = classes(winner);
             end
        end


    otherwise
        sprintf('force_membership(): <%s> is not a known template type.\n',par.template_type);

end
end


% =========================================================================
%  'center' two-pass helpers (port of force_membership.py)
% =========================================================================
function lbl = assign_to_templates(wf_out, templates, sd_scaled, members_wf, members_cls, amp_pct)
% Nearest-template assignment within a per-cluster distance radius, plus an
% optional amplitude gate. sd_scaled is the already-scaled radius per template.
n_out = size(wf_out,1);

% euclidean distances, n_out x K (no toolbox dependency)
G = bsxfun(@plus, sum(wf_out.^2,2), sum(templates.^2,2)') - 2*(wf_out*templates');
D = sqrt(max(G,0));

within = bsxfun(@lt, D, sd_scaled(:)');

if ~isempty(amp_pct)
    [peak_samples, amp_lo, amp_hi] = compute_amp_bounds(members_wf, members_cls, templates, amp_pct);
    cand_amps  = wf_out(:, peak_samples);            % n_out x K
    within_amp = bsxfun(@ge, cand_amps, amp_lo(:)') & bsxfun(@le, cand_amps, amp_hi(:)');
    within     = within & within_amp;
end

masked = D;
masked(~within) = inf;
[best_dist, best_idx] = min(masked, [], 2);
lbl = zeros(1, n_out);
ok  = best_dist < inf;
lbl(ok) = best_idx(ok);
end


function [peak_samples, amp_lo, amp_hi] = compute_amp_bounds(wf_in, cls_in, templates, pct_range)
% Per-cluster amplitude rejection bounds at each cluster's peak sample (index
% of largest |template| value). Clusters with <2 members get +/-inf (no-op).
[K, n_samples] = size(templates);
[~, peak_samples] = max(abs(templates), [], 2);
peak_samples = peak_samples(:)';
amp_lo = -inf(1, K);
amp_hi =  inf(1, K);
low_pct  = pct_range(1);
high_pct = pct_range(2);
cls_in = cls_in(:)';
for k = 1:K
    m = (cls_in == k);
    if sum(m) < 2
        continue
    end
    s = peak_samples(k);
    if s < 1 || s > n_samples
        continue
    end
    a = wf_in(m, s);
    amp_lo(k) = local_prctile_linear(a, low_pct);
    amp_hi(k) = local_prctile_linear(a, high_pct);
end
end


function out = mahalanobis_reassign(core_cls, core_feat, feat_out, labels)
% Keep the euclidean accept/reject in 'labels', then move every ACCEPTED spike
% to the cluster it is closest to in (shrunk) Mahalanobis. Rejected stay 0.
out = labels;
accepted = labels > 0;
if ~any(accepted)
    return
end
[means, inv_covs] = feature_models(core_cls, core_feat);
K = size(means, 1);
if K == 0
    return
end
M = feat_mahal(feat_out, means, inv_covs);           % n x K squared mahalanobis
[~, best] = min(M, [], 2);
best = best(:)';
out(accepted) = best(accepted);
end


function [means, inv_covs] = feature_models(cls, feats)
% Per-cluster mean + shrunk inverse covariance:
%   cov_s = 0.7*cov + 0.3*diag(cov) + 1e-6*I   (pinv guards the rest)
cls = cls(:)';
K = 0;
if ~isempty(cls) && max(cls) > 0
    K = double(max(cls));
end
F = size(feats, 2);
means    = zeros(K, F);
inv_covs = zeros(F, F, K);
for k = 1:K
    X = feats(cls == k, :);
    if isempty(X)
        inv_covs(:,:,k) = eye(F);
        continue
    end
    means(k, :) = mean(X, 1);
    if size(X,1) > F + 2
        C = cov(X);                              % N-1 denominator
    else
        C = diag(var(X, 0, 1) + 1e-6);           % N-1; degenerate -> diagonal
    end
    C = 0.7*C + 0.3*diag(diag(C)) + 1e-6*eye(F);
    inv_covs(:,:,k) = pinv(C);
end
end


function M = feat_mahal(feat_out, means, inv_covs)
% Squared Mahalanobis distance of each spike to each cluster, n x K.
K = size(means, 1);
n = size(feat_out, 1);
M = zeros(n, K);
for k = 1:K
    d = bsxfun(@minus, feat_out, means(k, :));
    M(:, k) = sum((d * inv_covs(:,:,k)) .* d, 2);
end
end


function q = local_prctile_linear(x, p)
% numpy-style percentile (method='linear'), so results match force_membership.py
x = sort(x(:));
n = numel(x);
if n == 0
    q = NaN; return
elseif n == 1
    q = x(1); return
end
pos  = p/100 * (n - 1);          % 0-indexed position
lo   = floor(pos);
hi   = ceil(pos);
frac = pos - lo;
q = x(lo+1)*(1-frac) + x(hi+1)*frac;
end


function v = local_getdef(s, f, d)
if isfield(s, f) && ~isempty(s.(f))
    v = s.(f);
else
    v = d;
end
end
