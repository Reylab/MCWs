function G = arm_wave_groups(T, cls, gt_t, tol)
% ARM_WAVE_GROUPS  Every waveform group the validation figures draw for one
% arm (Python-layout times file T, matched class CLS), plus the two
% reference templates every waveform panel carries.
%
%   gt_t - GT spike times of the GT class CLS was matched to
%   tol  - GT match tolerance (ms)
%
% G.templates      - matched template (mean of every arm spike in CLS that
%                    matches a GT spike) and the sd1 template (II) of CLS
% G.comp           - the pie's groups: I, II-I, III-II, IV (class CLS)
% G.c0             - cluster 0, never quarantined
% G.qnr            - quarantined, never rescued
% G.resc_by_prop   - every rescued spike, one group per quarantine property
% G.qnr_by_prop    - quarantined-never-rescued, one group per quarantine property
% G.per_class      - the desired class CLS only: .class, .templates (matched
%                    template + its sd1 template), .resc_by_prop (spikes
%                    rescued into CLS, one group per property)
% G.counts         - pie counts (n_I, n_core, n_II, n_III_II, n_IV, n_total,
%                    n_core_moved, n_sd1_core)
% G.M / G.prop_names / G.prop_active - per-spike flags for every potential
%                    quarantine mask, and which ones fed this run's quarantine
% G.Mx / G.prop_names_x / G.prop_active_x - the same, EXCLUSIVE: one column
%                    per mask (spikes that failed only it) + 'multiple masks'
%                    (failed 2+), so every breakdown adds up to its total
% G.masks          - logical row masks: in_cls, resc, q, is_c0, is_qnr, and the
%                    pie groups is_I, is_core (II-I), is_III (III-II), is_IV
% G.venn_rows      - arm-file rows of each cluster-venn region (S.rows)
% G.has_mahal / G.masks.is_mahal - only when the times file has per-spike
%                    mahal_reassigned (class set by the Mahalanobis step, not the
%                    template match); otherwise has_mahal = false -- never estimated
% Each group is a struct with .name, .W (n x samples), .color.

    lab = T.cluster_class(:,1);
    if isfield(T.quarantine, 'rescued')
        resc = logical(T.quarantine.rescued(:));
    else
        resc = false(size(T.cluster_class, 1), 1);   % no rescue step (clustering-only run)
    end
    forced = logical(T.forced(:));
    q = logical(T.quarantine.all(:));
    W = double(T.spikes);
    in_cls = lab == cls;

    [tmpl_sd1, ~, info] = sd1_pass_templates(T, cls, 1, [1 99]);
    core = false(size(lab)); core(info.clu_core_rows) = true;
    S = three_set_membership(gt_t, lab, T.cluster_class(:,2), q, in_cls, tol);
    tmpl_matched = mean(W([S.rows.gt_arm; S.rows.gt_arm_q], :), 1);
    G.templates = struct('name', {'matched template (GT & MCWs)', sprintf('sd1 template (II), class %d', cls)}, ...
                         'wave', {tmpl_matched, tmpl_sd1}, ...
                         'color', {[0 0 0], [0.85 0.1 0.6]}, 'style', {'-', '--'});

    is_I    = in_cls & ~forced & ~resc;
    is_core = in_cls & forced & core;
    is_III  = in_cls & forced & ~core;
    is_IV   = in_cls & resc;
    is_c0   = lab == 0 & ~q;
    is_qnr  = q & ~resc;

    G.comp = [grp('original SPC (I)', W(is_I,:), [0.05 0.35 0.12]), ...
              grp('sd1 first pass, not original (II-I)', W(is_core,:), [0.15 0.55 0.25]), ...
              grp('forced, sd3 (III-II)', W(is_III,:), [0.45 0.6 0.9]), ...
              grp('rescued (IV)', W(is_IV,:), [0.95 0.6 0.1])];
    G.c0  = grp('cluster 0, never quarantined', W(is_c0,:), [0.45 0.45 0.45]);
    G.qnr = grp('quarantined, never rescued', W(is_qnr,:), [0.95 0.6 0.1]);


    [M, names, active] = quarantine_property_flags(T);
    G.M = M; G.prop_names = names; G.prop_active = active;
    % Exclusive assignment for every breakdown: a spike that failed exactly
    % one active mask goes to that mask, a spike that failed 2+ goes to
    % 'multiple masks' -- so each breakdown's groups add up to its total.
    n_fail = sum(M(:, active), 2);
    G.Mx = [M & n_fail == 1, n_fail >= 2];
    G.prop_names_x = [names, {'multiple masks'}];
    G.prop_active_x = [active, true];
    G.resc_by_prop = grp('', zeros(0, size(W,2)), [0 0 0]); G.resc_by_prop(:) = [];
    G.qnr_by_prop  = G.resc_by_prop;
    for p = 1:numel(G.prop_names_x)
        nm = G.prop_names_x{p};
        if ~G.prop_active_x(p), nm = [nm ' -- off in this run']; end
        G.resc_by_prop(end+1) = grp(nm, W(resc & G.Mx(:,p), :), [0.1 0.6 0.7]);
        G.qnr_by_prop(end+1)  = grp(nm, W(is_qnr & G.Mx(:,p), :), [0.95 0.6 0.1]);
    end

    % the desired class: spikes rescued into it, by quarantine property
    gp = G.resc_by_prop([]);
    for p = 1:numel(G.prop_names_x)
        nm = G.prop_names_x{p};
        if ~G.prop_active_x(p), nm = [nm ' -- off in this run']; end
        gp(end+1) = grp(nm, W(is_IV & G.Mx(:,p), :), [0.1 0.6 0.7]); %#ok<AGROW>
    end
    G.per_class = struct('class', cls, 'templates', G.templates, 'resc_by_prop', gp);

    % how each forced / rescued spike got its class, ONLY when the run saved it
    % (per-spike mahal_reassigned, newer Python second pass). Never estimated.
    if isfield(T, 'mahal_reassigned') && numel(T.mahal_reassigned) == numel(lab)
        is_mahal = logical(T.mahal_reassigned(:));
        G.has_mahal = true;
    else
        is_mahal = false(size(lab));
        G.has_mahal = false;
    end

    G.counts = struct('n_I', nnz(is_I), 'n_core', nnz(is_core), 'n_II', nnz(is_I) + nnz(is_core), ...
        'n_III_II', nnz(is_III), 'n_IV', nnz(is_IV), 'n_total', nnz(in_cls), ...
        'n_core_moved', nnz(core & ~in_cls), 'n_sd1_core', info.clu_core);
    G.lab = lab;
    G.masks = struct('in_cls', in_cls, 'resc', resc, 'q', q, 'is_c0', is_c0, 'is_qnr', is_qnr, ...
        'is_I', is_I, 'is_core', is_core, 'is_III', is_III, 'is_IV', is_IV, ...
        'is_mahal', is_mahal);
    % cluster-venn region of every arm spike in CLS (three_set_membership:
    % GT matched to the arm first, then split by quarantine)
    G.venn_rows = S.rows;
end

function g = grp(name, W, color)
    g = struct('name', name, 'W', W, 'color', color);
end
