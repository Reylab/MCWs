function subsets = save_validation_subsets(cfg, arm_classes)
% SAVE_VALIDATION_SUBSETS  Save the spike subsets the validation figures are
% built from, one file per arm, so any of them can be pulled up again
% (waveforms, times, rows) or rastered (plot_subset_rasters) without rerunning.
%
%   save_validation_subsets(cfg)               % matches arm classes itself
%   save_validation_subsets(cfg, arm_classes)  % reuse run_validation's matching
%
% Subsets (cluster venn, GT class vs the arm's matched class -- the same
% three_set_membership regions as the venn page, and the composition pie's
% groups from arm_wave_groups):
%   GT side (GT-file rows, GT waveforms)
%     gt              every GT spike of cfg.gt.class
%     gt_only         GT not in arm, not in quarantine
%     gt_lost         GT not in arm, in quarantine (never rescued into the arm)
%     gt_not_in_arm   gt_only + gt_lost
%   arm side (arm-file rows, arm waveforms)
%     arm             every arm spike of the matched class
%     matched         GT & arm (rescued or not)
%     matched_rescued GT & arm, came in by rescue
%     arm_only        arm not in GT, not rescued
%     arm_rescued_not_gt  arm not in GT, rescued
%     arm_not_in_gt   arm_only + arm_rescued_not_gt
%     rescued         every rescued spike in the class (IV)
%     pie_I / pie_II_I / pie_III_II / pie_IV   composition pie groups
%     q_not_rescued   every quarantined spike on the channel never rescued
%   channel-wide, only with cfg.channel_subsets = true:
%     det_*           every GT spike vs every arm spike (any class incl. 0), the
%                     detections venn -- det_gt_only (GT rows), det_arm_only,
%                     det_both (arm rows). Detection differences only: no
%                     quarantine split and no quarantine fields.
%     qnr_<property>  q_not_rescued split by the quarantine property it failed,
%                     exclusive as on the waveform breakdown page (one mask, or
%                     qnr_multiple_masks for 2+); active masks with spikes only
%
% Each subset also has, per spike: .own_class (its class in its own file, 0
% included), .partner_row / .partner_class (the one-to-one time-matched spike in
% the OTHER file -- the arm for GT subsets, GT for arm subsets -- and its class,
% 0 included, NaN = not detected there), and .quarantined/.rescued (arm rows) or
% .partner_quarantined (GT rows). subset_partner_classes.csv tallies partner_class
% per subset: where each subset's spikes ended up in the other file.
%
% Each subset: .name .side ('gt'|'arm') .desc .rows (rows of its times file)
% .times (ms) .spikes (n x samples) .class (the class it was taken from)
%
% OUTPUTS (under cfg.out_dir/subsets)
%   subsets_<arm>.mat - struct `subsets` (one field per subset) + cfg/arm_class
%   subset_counts.csv - n per (arm, subset)

if nargin < 2, arm_classes = []; end
[Gs, ~, times_arms, arm_classes] = prepare_arm_groups(cfg, arm_classes);
times_gt = load(fullfile(cfg.gt.dir, cfg.gt.times_file), 'cluster_class', 'spikes');
gt_cc = times_gt.cluster_class;
gt_rows = find(gt_cc(:,1) == cfg.gt.class);

out_dir = fullfile(cfg.out_dir, 'subsets');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

rows_csv = {}; tally_all = {};
for k = 1:numel(cfg.arms)
    T = times_arms{k};
    G = Gs{k};
    nm = cfg.arms(k).name;
    cls = arm_classes(k);
    lab = T.cluster_class(:,1);
    S = three_set_membership(gt_cc(gt_rows, 2), lab, T.cluster_class(:,2), ...
        logical(T.quarantine.all(:)), lab == cls, cfg.tol);
    R = S.rows;
    m = G.masks;
    % every GT spike <-> every arm spike, one-to-one within tol (as the
    % detections venn): each subset spike's class in the OTHER file
    [gs, go] = sort(gt_cc(:, 2));
    gm0 = nearest_match(gs, T.cluster_class(:, 2), cfg.tol);
    gmu0 = enforce_one_to_one(gs, T.cluster_class(:, 2), gm0, cfg.tol);
    gt2arm = zeros(size(gt_cc, 1), 1); gt2arm(go(gmu0 > 0)) = gmu0(gmu0 > 0);
    arm2gt = zeros(numel(lab), 1);     arm2gt(gmu0(gmu0 > 0)) = go(gmu0 > 0);
    qall = logical(T.quarantine.all(:)); rsc = false(numel(lab), 1);
    if isfield(T.quarantine, 'rescued'), rsc = logical(T.quarantine.rescued(:)); end

    g = @(name, rows, desc) mk(name, 'gt', desc, sort(rows(:)), gt_cc, times_gt.spikes, cfg.gt.class);
    a = @(name, rows, desc) mk(name, 'arm', desc, sort(rows(:)), T.cluster_class, T.spikes, cls);
    L = {
        g('gt',              gt_rows,                              sprintf('%s class %d', cfg.gt.name, cfg.gt.class))
        g('gt_only',         gt_rows(S.gt_only_pos),               sprintf('%s not in %s, not quarantined', cfg.gt.name, nm))
        g('gt_lost',         gt_rows(S.gt_lost_pos),               sprintf('%s not in %s, quarantined (lost)', cfg.gt.name, nm))
        g('gt_not_in_arm',   gt_rows([S.gt_only_pos; S.gt_lost_pos]), sprintf('%s not in %s', cfg.gt.name, nm))
        a('arm',             find(m.in_cls),                       sprintf('%s class %d', nm, cls))
        a('matched',         [R.gt_arm; R.gt_arm_q],               sprintf('%s & %s matched', cfg.gt.name, nm))
        a('matched_rescued', R.gt_arm_q,                           sprintf('%s & %s matched, rescued', cfg.gt.name, nm))
        a('arm_only',        R.arm_only,                           sprintf('%s not in %s, not rescued', nm, cfg.gt.name))
        a('arm_rescued_not_gt', R.arm_q,                           sprintf('%s not in %s, rescued', nm, cfg.gt.name))
        a('arm_not_in_gt',   [R.arm_only; R.arm_q],                sprintf('%s not in %s', nm, cfg.gt.name))
        a('rescued',         find(m.is_IV),                        sprintf('rescued into %s class %d (IV)', nm, cls))
        a('pie_I',           find(m.is_I),                         'original SPC (I)')
        a('pie_II_I',        find(m.is_core),                      'sd1 first pass, not original (II-I)')
        a('pie_III_II',      find(m.is_III),                       'forced, sd3 (III-II)')
        a('pie_IV',          find(m.is_IV),                        'rescued (IV)')
        a('q_not_rescued',   find(m.is_qnr),                       'quarantined, never rescued (channel)')};
    for pi_ = find(G.prop_active_x)
        r = find(m.is_qnr & G.Mx(:, pi_));
        if isempty(r), continue; end
        L{end+1} = a(['qnr_' matlab.lang.makeValidName(G.prop_names_x{pi_})], r, ...
            sprintf('quarantined, never rescued: %s', G.prop_names_x{pi_})); %#ok<AGROW>
    end
    if isfield(cfg, 'channel_subsets') && cfg.channel_subsets
        % detections venn: every spike in both files, one-to-one within tol
        gF = @(name, rows, desc) mk(name, 'gt', desc, sort(rows(:)), gt_cc, times_gt.spikes, NaN);
        aF = @(name, rows, desc) mk(name, 'arm', desc, sort(rows(:)), T.cluster_class, T.spikes, NaN);
        L = [L; {
            gF('det_gt_only',          find(gt2arm == 0), sprintf('every %s spike not detected by %s', cfg.gt.name, nm))
            aF('det_arm_only',         find(arm2gt == 0), sprintf('every %s spike not detected by %s', nm, cfg.gt.name))
            aF('det_both',             find(arm2gt > 0), sprintf('detected by both %s and %s (arm rows)', cfg.gt.name, nm))}];
    end
    subsets = struct(); tally = {};
    for i = 1:numel(L)
        L{i} = add_classes(L{i}, gt_cc, lab, gt2arm, arm2gt, qall, rsc, ~startsWith(L{i}.name, 'det_'));
        subsets.(L{i}.name) = L{i};
        pcs = L{i}.partner_class; pcs(isnan(pcs)) = -1;
        for v = unique(pcs)'
            tally(end+1, :) = {nm, cls, L{i}.name, L{i}.side, ...
                ifelse(v < 0, 'not detected', sprintf('class %d', v)), nnz(pcs == v)}; %#ok<AGROW>
        end
        rows_csv(end+1, :) = {nm, cls, L{i}.name, L{i}.side, numel(L{i}.rows), L{i}.desc}; %#ok<AGROW>
    end
    arm_class = cls; %#ok<NASGU>
    tally_all = [tally_all; tally]; %#ok<AGROW>
    save(fullfile(out_dir, sprintf('subsets_%s.mat', nm)), 'subsets', 'cfg', 'arm_class', '-v7.3');
    fprintf('[subsets] %s class %d: gt_only %d, gt_lost %d, matched %d, arm_only %d, arm_rescued_not_gt %d\n', ...
        nm, cls, numel(subsets.gt_only.rows), numel(subsets.gt_lost.rows), numel(subsets.matched.rows), ...
        numel(subsets.arm_only.rows), numel(subsets.arm_rescued_not_gt.rows));
end
writetable(cell2table(tally_all, 'VariableNames', {'arm','arm_class','subset','side','partner_class','n'}), ...
    fullfile(out_dir, 'subset_partner_classes.csv'));
writetable(cell2table(rows_csv, 'VariableNames', {'arm','arm_class','subset','side','n','desc'}), ...
    fullfile(out_dir, 'subset_counts.csv'));
end

function s = add_classes(s, gt_cc, lab, gt2arm, arm2gt, qall, rsc, with_q)
% with_q = false for the detection subsets: those compare every spike in both
% files, so no quarantine fields
% each subset spike's class in its own file and in the other file (0 = unsorted,
% NaN = the other file has no spike there), plus quarantine flags for arm rows
    r = s.rows(:);
    if strcmp(s.side, 'gt')
        s.own_class = gt_cc(r, 1); s.partner_row = gt2arm(r);
        pc = nan(numel(r), 1); k = s.partner_row > 0; pc(k) = lab(s.partner_row(k));
        s.partner_class = pc;
        if with_q, s.partner_quarantined = false(numel(r), 1); s.partner_quarantined(k) = qall(s.partner_row(k)); end
    else
        s.own_class = lab(r); s.partner_row = arm2gt(r);
        pc = nan(numel(r), 1); k = s.partner_row > 0; pc(k) = gt_cc(s.partner_row(k), 1);
        s.partner_class = pc;
        if with_q, s.quarantined = qall(r); s.rescued = rsc(r); end
    end
end

function v = ifelse(c, a, b), if c, v = a; else, v = b; end, end

function s = mk(name, side, desc, rows, cc, spikes, cls)
    s = struct('name', name, 'side', side, 'desc', desc, 'rows', rows, ...
        'times', cc(rows, 2), 'spikes', double(spikes(rows, :)), 'class', cls);
end
