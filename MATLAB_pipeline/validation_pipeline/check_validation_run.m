function R = check_validation_run(cfg)
% CHECK_VALIDATION_RUN  Sanity report for one finished run_validation output.
%
%   R = check_validation_run(validation_cfg_pt9_emu3_ch331())
%
% Prints, and returns in R:
%   GT class        present?  n in the times file vs n in the grapes file
%                   (grapes.rasters.chanN.details.classN), both values when
%                   they differ
%   each arm        matched class, n in times vs grapes (and the pre-rescue
%                   classN + classN_rescue counts the rescue grapes carries)
%   venn (cluster)  gt_only + gt_q + matched = n_gt; matched + arm_only +
%                   arm_q = n_arm
%   pie             II (incl. I) + (III-II) + IV = cluster n
%   subsets         counts equal the venn regions; qnr_<property> sum to
%                   q_not_rescued = quarantined - rescued (any class)
%   clustered arm   (no rescue step) has 0 rescued
% Problems are collected in R.issues (empty = all checks passed).

out = cfg.out_dir;
R = struct('channel', cfg.ch, 'gt_class', cfg.gt.class, 'issues', {{}}, 'arms', struct([]));
iss = @(varargin) sprintf(varargin{:});
cf = sprintf('chan%d', cfg.ch);

% --- GT class: times vs grapes
Tg = load(fullfile(cfg.gt.dir, cfg.gt.times_file), 'cluster_class');
Gg = load(fullfile(cfg.gt.dir, cfg.gt.grapes_file), 'rasters');
R.gt_times_n = nnz(Tg.cluster_class(:,1) == cfg.gt.class);
R.gt_times_classes = max(Tg.cluster_class(:,1));
R.gt_grapes_n = grapes_count(Gg, cf, cfg.gt.class);
R.gt_grapes_classes = numel(class_fields(Gg, cf));
fprintf('\n=== ch%d  (%s)\n', cfg.ch, out);
fprintf('GT class %d: times n=%d, grapes n=%s   [times has classes 1..%d, grapes has %d class fields]\n', ...
    cfg.gt.class, R.gt_times_n, num2str(R.gt_grapes_n), R.gt_times_classes, R.gt_grapes_classes);
if R.gt_times_n == 0, R.issues{end+1} = iss('GT class %d not in the times file', cfg.gt.class); end
if isnan(R.gt_grapes_n), R.issues{end+1} = iss('GT class %d not in the grapes file', cfg.gt.class);
elseif R.gt_grapes_n ~= R.gt_times_n
    R.issues{end+1} = iss('GT class %d: times %d vs grapes %d', cfg.gt.class, R.gt_times_n, R.gt_grapes_n);
end
if R.gt_times_classes ~= R.gt_grapes_classes
    R.issues{end+1} = iss('GT times has %d classes, grapes has %d', R.gt_times_classes, R.gt_grapes_classes);
end

% --- arms
ac = readtable(fullfile(out, 'arm_classes.csv'), 'TextType', 'char');
V = readtable(fullfile(out, 'venn_three_set', 'venn3_counts.csv'), 'TextType', 'char');
C = readtable(fullfile(out, 'cluster_composition', 'composition_counts.csv'), 'TextType', 'char');
for k = 1:numel(cfg.arms)
    A = cfg.arms(k);
    cls = ac.class(strcmp(ac.arm, A.name));
    T = load(fullfile(A.dir, A.times_file), 'cluster_class', 'quarantine');
    G = load(fullfile(A.dir, A.grapes_file), 'rasters');
    lab = T.cluster_class(:,1);
    a.name = A.name; a.class = cls;
    a.times_n = nnz(lab == cls);
    a.grapes_n = grapes_count(G, cf, cls);
    a.grapes_pre = grapes_count(G, cf, cls, '');          % same field; rescue grapes keeps pre-rescue
    a.grapes_resc = grapes_count(G, cf, cls, '_rescue');  % delta field, if any
    a.n_rescued = 0;
    if isfield(T.quarantine, 'rescued'), a.n_rescued = nnz(T.quarantine.rescued); end
    a.n_rescued_cls = 0;
    if isfield(T.quarantine, 'rescued'), a.n_rescued_cls = nnz(T.quarantine.rescued(:) & lab == cls); end
    a.class_sizes = arrayfun(@(c) nnz(lab == c), 0:max(lab));
    fprintf('%s -> class %d: times n=%d (rescued into it %d), grapes n=%s', A.name, cls, a.times_n, ...
        a.n_rescued_cls, num2str(a.grapes_n));
    if ~isnan(a.grapes_resc), fprintf(' (+%s in class%d_rescue)', num2str(a.grapes_resc), cls); end
    fprintf('\n   class sizes 0..%d: %s\n', max(lab), mat2str(a.class_sizes));
    g_total = a.grapes_n; if ~isnan(a.grapes_resc), g_total = a.grapes_n + a.grapes_resc; end
    if isnan(a.grapes_n), R.issues{end+1} = iss('%s class %d not in its grapes file', A.name, cls);
    elseif g_total ~= a.times_n && a.grapes_n ~= a.times_n
        R.issues{end+1} = iss('%s class %d: times %d vs grapes %d (+rescue %s)', A.name, cls, a.times_n, ...
            a.grapes_n, num2str(a.grapes_resc));
    end
    if contains(A.name, 'clustered') && a.n_rescued > 0
        R.issues{end+1} = iss('%s is clustering-only but has %d rescued', A.name, a.n_rescued);
    end

    % venn (cluster scope)
    v = V(strcmp(V.arm, A.name) & strcmp(V.scope, 'cluster'), :);
    if ~isempty(v)
        okg = v.gt_only + v.gt_q + v.gt_arm + v.gt_arm_q == v.n_gt;
        oka = v.gt_arm + v.gt_arm_q + v.arm_only + v.arm_q == v.n_arm;
        fprintf('   venn: n_gt %d = %d+%d+%d+%d %s | n_arm %d = %d+%d+%d+%d %s | quarantined %d, rescued into class %d\n', ...
            v.n_gt, v.gt_only, v.gt_q, v.gt_arm, v.gt_arm_q, tf(okg), v.n_arm, v.gt_arm, v.gt_arm_q, ...
            v.arm_only, v.arm_q, tf(oka), v.n_quarantined, v.gt_arm_q + v.arm_q);
        if ~okg || ~oka, R.issues{end+1} = iss('%s venn regions do not add up', A.name); end
        if v.n_gt ~= R.gt_times_n, R.issues{end+1} = iss('%s venn n_gt %d ~= GT times %d', A.name, v.n_gt, R.gt_times_n); end
        if v.n_arm ~= a.times_n, R.issues{end+1} = iss('%s venn n_arm %d ~= times %d', A.name, v.n_arm, a.times_n); end
        if v.gt_arm_q + v.arm_q ~= a.n_rescued_cls
            R.issues{end+1} = iss('%s venn rescued %d ~= rescued into class %d', A.name, v.gt_arm_q + v.arm_q, a.n_rescued_cls);
        end
        a.recall = (v.gt_arm + v.gt_arm_q) / v.n_gt;
        a.precision = (v.gt_arm + v.gt_arm_q) / v.n_arm;
        fprintf('   matched %d: %.1f%% of GT, %.1f%% of arm\n', v.gt_arm + v.gt_arm_q, 100*a.recall, 100*a.precision);
    end

    % pie
    c = C(strcmp(C.arm, A.name), :);
    if ~isempty(c)
        vn = c.Properties.VariableNames;
        parts = intersect({'n_II_sd1_first_pass','n_III_minus_II_forced','n_IV_rescued'}, vn, 'stable');
        s = sum(c{1, parts});
        tot = c.n_total;
        fprintf('   pie: %s = %d of %d %s\n', strjoin(compose('%s %d', string(parts'), c{1, parts}'), ' + '), s, tot, tf(s == tot));
        if s ~= tot, R.issues{end+1} = iss('%s pie groups %d ~= cluster %d', A.name, s, tot); end
        if tot ~= a.times_n, R.issues{end+1} = iss('%s pie total %d ~= times %d', A.name, tot, a.times_n); end
    end

    % subsets
    sf = fullfile(out, 'subsets', sprintf('subsets_%s.mat', A.name));
    if exist(sf, 'file') && ~isempty(v)
        S = load(sf, 'subsets'); s = S.subsets; n = @(f) numel(s.(f).rows);
        pairs = {'gt_only', v.gt_only; 'gt_lost', v.gt_q; 'matched', v.gt_arm + v.gt_arm_q; ...
                 'arm_only', v.arm_only; 'arm_rescued_not_gt', v.arm_q};
        bad = pairs(cellfun(@(f, x) n(f) ~= x, pairs(:,1), pairs(:,2)), 1);
        fn = fieldnames(s); qn = fn(startsWith(fn, 'qnr_'));
        qsum = sum(cellfun(n, qn));
        q_exp = v.n_quarantined - a.n_rescued;
        fprintf('   subsets vs venn %s | q_not_rescued %d = quarantined %d - rescued %d %s | qnr properties sum %d %s\n', ...
            tf(isempty(bad)), n('q_not_rescued'), v.n_quarantined, a.n_rescued, tf(n('q_not_rescued') == q_exp), ...
            qsum, tf(qsum == n('q_not_rescued')));
        if ~isempty(bad), R.issues{end+1} = iss('%s subsets differ from venn: %s', A.name, strjoin(bad, ', ')); end
        if n('q_not_rescued') ~= q_exp, R.issues{end+1} = iss('%s q_not_rescued %d ~= %d', A.name, n('q_not_rescued'), q_exp); end
        if qsum ~= n('q_not_rescued'), R.issues{end+1} = iss('%s qnr properties sum %d ~= %d', A.name, qsum, n('q_not_rescued')); end
        % channel-wide subsets (cfg.channel_subsets): full and detections venn regions
        vf = V(strcmp(V.arm, A.name) & strcmp(V.scope, 'full'), :);
        if isfield(s, 'full_gt_only') && ~isempty(vf)
            pf = {'full_gt_only', vf.gt_only; 'full_gt_lost', vf.gt_q; 'full_matched', vf.gt_arm + vf.gt_arm_q; ...
                  'full_arm_only', vf.arm_only; 'full_arm_rescued_not_gt', vf.arm_q};
            badf = pf(cellfun(@(f, x) n(f) ~= x, pf(:,1), pf(:,2)), 1);
            fprintf('   full subsets vs full venn %s\n', tf(isempty(badf)));
            if ~isempty(badf), R.issues{end+1} = iss('%s full subsets differ from venn: %s', A.name, strjoin(badf, ', ')); end
        end
        vd = V(strcmp(V.arm, A.name) & strcmp(V.scope, 'detections'), :);
        if isfield(s, 'det_both') && ~isempty(vd)
            okd = n('det_both') == vd.gt_arm && n('det_gt_only') == vd.gt_only && n('det_arm_only') == vd.arm_only;
            fprintf('   detection subsets vs detections venn %s (both %d, gt only %d, arm only %d)\n', tf(okd), n('det_both'), n('det_gt_only'), n('det_arm_only'));
            if ~okd, R.issues{end+1} = iss('%s detection subsets differ from the detections venn', A.name); end
        end
    end
    R.arms = [R.arms, a];
end

% raster page title counts: spikes of each unit inside the drawn windows
st = load(fullfile(out, 'validation_state.mat'));
units = [{cfg.gt.name}, {cfg.arms.name}];
cls_u = [cfg.gt.class; ac.class(:)];
files = [{fullfile(cfg.gt.dir, cfg.gt.times_file)}, arrayfun(@(A) fullfile(A.dir, A.times_file), cfg.arms, 'UniformOutput', false)];
R.title_n = nan(1, numel(units));
for u = 1:numel(units)
    if numel(st.ref_tables) < u, break; end
    d = st.ref_tables{u};
    T = load(files{u}, 'cluster_class');
    n = count_in_windows(T.cluster_class(T.cluster_class(:,1) == cls_u(u), 2), d.stim_number, d.win_lo, d.win_hi, st.onset_map);
    R.title_n(u) = n;
    saved = NaN; if isfield(st, 'n_in_window'), saved = st.n_in_window(u); end
    fprintf('raster title %s class %d: %d in window (page used %s) %s\n', units{u}, cls_u(u), n, num2str(saved), tf(n == saved && n > 0));
    if n == 0, R.issues{end+1} = iss('%s raster title count is 0', units{u}); end
    if ~isnan(saved) && n ~= saved, R.issues{end+1} = iss('%s raster title %d ~= recount %d', units{u}, saved, n); end
    if isnan(saved), R.issues{end+1} = iss('%s: no saved title count (rerun rasters)', units{u}); end
end

% outputs that should exist
need = {'stimulus_rasters', 'venn_three_set', 'cluster_composition', 'pie_venn', 'waveform_breakdowns', ...
        'subsets', 'results', 'response_waveforms'};   % subset_rasters is optional (only when asked for)
miss = need(~cellfun(@(d) exist(fullfile(out, d), 'dir') == 7, need));
if ~isempty(miss), R.issues{end+1} = ['missing outputs: ' strjoin(miss, ', ')]; end

if isempty(R.issues)
    fprintf('ALL CHECKS PASSED\n');
else
    fprintf('ISSUES:\n'); fprintf('  - %s\n', R.issues{:});
end
end

function n = grapes_count(G, cf, cls, suffix)
    if nargin < 4, suffix = ''; end
    n = NaN;
    f = sprintf('class%d%s', cls, suffix);
    if isfield(G.rasters, cf) && isfield(G.rasters.(cf), 'details') && isfield(G.rasters.(cf).details, f)
        n = G.rasters.(cf).details.(f);
    end
end

function f = class_fields(G, cf)
    f = {};
    if isfield(G.rasters, cf)
        f = fieldnames(G.rasters.(cf));
        f = f(~cellfun(@isempty, regexp(f, '^class\d+$', 'once')));
    end
end

function s = tf(ok)
    if ok, s = 'OK'; else, s = 'MISMATCH'; end
end
