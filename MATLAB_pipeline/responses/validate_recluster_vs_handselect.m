t function out = validate_recluster_vs_handselect(channels, varargin)
% VALIDATE_RECLUSTER_VS_HANDSELECT  Measure how the REAL pipeline (artifact
% rejection -> Do_features -> Do_clustering, then rescue) changes stimulus
% responses relative to a HAND-SELECTED ground truth, WITH the clustering
% allowed to change.
%
% This is the complement of validate_mask_effects.m. That script freezes the
% clustering and only adds/removes spikes from fixed clusters, so it hides the
% fact that within_channel + refract_viol change the wavelet feature space and
% therefore change SPC (cluster count / membership / shape). Here every arm
% re-runs SPC for real, and each arm's classes are matched back to the
% hand-selected class by spike-time overlap before any response metric is
% compared - hand-select is the ground truth and every comparison is
% class-vs-class against it.
%
% Arms (each compared against 'handselect'):
%   handselect              the hand-select grapes*.mat is the GROUND TRUTH and
%                           is scored AS-IS (never rebuilt). Its clustering, for
%                           class matching, is RECONSTRUCTED from that raster's
%                           spike times (see reconstruct_cc_from_grapes) - so
%                           the grapes is the only thing that has to be right.
%   nomask_recluster        Do_features + Do_clustering fresh, NO masks
%                           (isolates re-SPC non-determinism).
%   masked_recluster        within_channel + refract_viol, then Do_features +
%                           Do_clustering fresh (isolates the rejection /
%                           feature-space effect).
%   masked_recluster_rescue masked_recluster + rescue_spikes (what rescue
%                           recovers).
%
% Per recluster arm: snapshot spikes+times+templates into <out>/<arm>/, build
% grapes (mu + sorted [+ rescue]), characterize_grapes_responses. Class matching
% is done on the grapes rasters of BOTH sides (post-onset spikes reconstructed
% to absolute time, same coverage for plain and blanks flavours) so recall /
% precision are apples-to-apples.
%
% Required
%   channels    numeric chan_ID vector
%
% Name-value
%   'handselect_dir'   ''    REQUIRED. Folder holding the ground-truth
%                            grapes*.mat (and optionally times_<name>.mat +
%                            <name>_spikes.mat, used only for cluster-quality
%                            metrics + the waveform overlay). Relative to the
%                            session root or absolute.
%   'handselect_grapes' ''   filename of the ground-truth raster inside
%                            handselect_dir. Default: the single grapes*.mat
%                            there (errors if 0 or >1 - keep only the one you
%                            want, or name it here).
%   'spikes_src'       ''    No-mask spikes folder (default: newest spikes*
%                            under the session root). Its *_spikes.mat are
%                            copied to a working folder; masks are applied to
%                            the copy, never the source.
%   'use_blanks'       true
%   'circshiftblanks'  false
%   'seed'             147   Do_clustering random seed, shared by every recluster
%                            arm so the comparison is reproducible. 0 = clock
%                            (non-deterministic - the production pipeline's
%                            default, but bad for a validation).
%   'clu_par'          struct() overrides for the Do_clustering par (default is
%                            byte-identical to processing_steps_MCW_reduced:
%                            min_clus 20, max_spk 30000, maxtemp 0.251,
%                            tempstep 0.01, max_std_templates 3). e.g.
%                            struct('maxtemp',0.32) lets automated SPC split
%                            further - the automatic temperature pick is
%                            conservative and under-segments vs a GUI operator.
%   'arms'  {'nomask_recluster','masked_recluster','masked_recluster_rescue'}
%   'within_channel_detections' 'all'   forwarded to within_channel('detections',..)
%   'refract_keep_strategy'     'none'  forwarded to refract_viol('keep_strategy',..)
%   'stim_list'        'all'  the stimulus selection to score - one of:
%                            'all'  = union of pics2use over every screening
%                                     block (broad; baseline spans everything)
%                            numeric vector of stimulus numbers
%                            (see 'scr' for "one input screening block")
%   'scr'             []      screening-block index (1..#SCR). When set,
%                            stim_list is REPLACED by scr_config_cell{scr}.pics2use
%                            - i.e. score exactly one input stimulus selection.
%                            The same list reaches every arm, so the zscore /
%                            mu_base / std_base baseline (computed across it)
%                            is identical across arms.
%   'exp_type'         'RSVP_online'
%   'gt_classes'       'all'  numeric vector of hand-select class numbers to
%                            report, or 'all'
%   'match_tol_ms'     0.5    spike-time match tolerance (ground truth spikes
%                            match the new detection to ~0.1 ms in practice)
%   'match_min_recall' 0.5    a ground-truth class is "matched" when the best
%                            arm class captures >= this fraction of its spikes
%   'resp_z_thr'       3      |zscore| cutoff (on top of the pipeline's own
%                            min_spk_test & p_test & good_lat gate) for calling
%                            a (class,stim) responsive
%   'assess_rejected'  true   build sub-rasters for the spikes the pipeline
%                            REJECTS (quarantine / refract) and RESCUES, and
%                            score their stimulus response
%   'do_sorting_metrics' true run compute_metrics_batch per arm (writes the
%                            metrics/ waveform+ISI+silhouette PNGs into every
%                            arm folder), stack the cluster-quality metrics
%                            across arms keyed by the MATCHED class, and draw a
%                            GT-vs-arm mean-waveform overlay per (channel, class)
%   'do_rasters'       true   run plot_grapes_as_online per arm into its folder
%                            (offline_grapes_new_order_results*). Slowest stage.
%   'order_by_rank'    false  raster stim-row order. false = fixed stimulus
%                            order, so stimulus N is on the SAME row in every
%                            arm's raster (what you want for a validation, esp.
%                            with a stim_list / scr). true = per-arm response
%                            rank (rows are then NOT comparable across arms).
%   'run_features'     true   run Do_features (set false to reuse existing)
%   'do_plots'         true
%   'out_dir'          ''     default <root>/RECLUSTER_VALIDATION_<ts>
%
% Output struct 'out'
%   .out_dir .params
%   .arms          struct array: name, dir, metrics (path), flavour
%   <out>/handselect/reconstructed_cc_<name>.mat  the cluster_class rebuilt
%                  from the ground-truth grapes (post-onset spikes, absolute ms)
%   <out>/handselect/times_<name>.mat  a REAL times file for that reconstruction
%                  - waveforms matched by spike time from the working spikes
%                  file, wavelet features recomputed for exactly that subset - so
%                  cluster-quality metrics + the waveform overlay run off the
%                  grapes clustering, not the (often stale) folder times file
%   .match.<arm>   table, one row per gt class: arm_class, n_gt, n_arm,
%                  n_shared, recall, precision, arm_class_dominant_gt, status,
%                  n_gt_unmatched, frac_unmatched (gt spikes with no arm
%                  counterpart), frac_elsewhere (gt spikes that landed in a
%                  DIFFERENT arm class), n_arm_from_other_gt, n_arm_new (spikes
%                  in the matched arm cluster with NO gt counterpart at all)
%   .match_detail        long: for each gt class, spike count -> every
%                  destination (arm classK / class0 / unmatched)
%                  -> comparisons/class_match_where_gt_spikes_go.csv
%   .match_composition   long: what each matched arm cluster is made of
%                  (this gt class / other gt / gt class0 / new)
%                  -> comparisons/class_match_arm_cluster_composition.csv
%   .compare       struct, field ch<NN>_gtclass<c> -> per-arm response matrices
%                  (+ match_recall / match_precision / match_frac_* per arm)
%   .summary       table: arm, channel_id, gt_class, arm_class, match_status,
%                  n_responsive, frac_responsive, n_gained, n_lost,
%                  jaccard_resp, mean_zscore
%   .assess        (assess_rejected) struct with the reject/rescue sub-raster
%                  metrics tables
%   .sorting_metrics (do_sorting_metrics) long table: arm, channel_id, gt_class,
%                  arm_class, num_spikes, snr, firing_rate, isi_violation_rate,
%                  isolation_distance, l_ratio, d_prime, ... (one row per
%                  arm x matched cluster) -> comparisons/sorting_metrics.csv
%
% Plots / report (do_plots)
%   comparisons/<key>_plots/  change_/heatmap_/heatmap_delta_/responsive_count/
%                             waveform_overlay per (channel, gt class)
%   comparisons/dashboard_<key>.png  one figure: match + response + cluster
%                             quality bars side by side, x = arm
%   comparisons/class_match_ch<NN>_<arm>.png  two stacked-bar panels: where
%                             each GT class's spikes go, and what each matched
%                             arm cluster is made of
%   comparisons/sorting_metrics_<key>.png  quality slope plot across arms
%   <arm>/temperature_<name>.png  +  comparisons/temperature_ch<NN>_<arm>.png
%                             stand-alone SPC temperature diagram from the tree
%                             (data_<name>.dg_01): cluster size vs T, and
%                             # clusters >= min_clus vs T (where SPC would
%                             split), with the auto-selected temperature marked.
%                             NOT the full wave_clus 3x5 figure.  (do_clu_plots)
%   comparisons/feature_space_ch<NN>.png + feature_space.csv  KS statistic per
%                             wavelet coefficient (ranked). Solid = each recluster
%                             arm's real Do_features KS (full population); dashed
%                             = handselect over the grapes response-window spikes,
%                             with and without class 0. Legend shows spike count.
%                             nomask = the hand-select all-spikes feature space
%                             (no masking on either); nomask->masked = the
%                             artifact-rejection effect.
%   comparisons/cluster_comparison_ch<NN>/  per arm: acg_ccg_<arm>.png (ACG on
%                             the diagonal, CCG off-diagonal - a lag-0 dip means
%                             two clusters are one unit), waveform_xcorr_<arm>.png
%                             (mean-waveform correlation matrix),
%                             spike_overlap_<arm>.png (GT class x arm class
%                             spike-time Jaccard - shows the many->one collapse)
%   report.html               big-scale page: per (channel, gt class) the
%                             match / composition / response / quality tables,
%                             the figures, and a stimulus grid (thumbnail +
%                             zscore per arm, colour-coded, responsive marked)
%
% Example
%   cd '.../new trial pt 6 emu 7 ch 334'
%   out = validate_recluster_vs_handselect(334, 'handselect_dir','hand selected', ...
%            'gt_classes',[1 2 3]);

    % ================= args =================
    p = inputParser;
    addParameter(p, 'handselect_dir', '', @ischar);
    addParameter(p, 'handselect_grapes', '', @ischar);   % '' = the single grapes*.mat in handselect_dir
    addParameter(p, 'spikes_src', '', @ischar);
    addParameter(p, 'use_blanks', true, @islogical);
    addParameter(p, 'circshiftblanks', false, @islogical);
    addParameter(p, 'seed', 147, @isnumeric);   % nonzero -> reproducible SPC
    addParameter(p, 'clu_par', struct(), @isstruct);   % override Do_clustering par fields
    addParameter(p, 'arms', {'nomask_recluster','masked_recluster','masked_recluster_rescue'}, @iscell);
    addParameter(p, 'within_channel_detections', 'all');
    addParameter(p, 'refract_keep_strategy', 'none', @ischar);
    addParameter(p, 'stim_list', 'all');
    addParameter(p, 'scr', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x)));
    addParameter(p, 'exp_type', 'RSVP_online', @ischar);
    addParameter(p, 'gt_classes', 'all');
    addParameter(p, 'match_tol_ms', 0.5, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'match_min_recall', 0.5, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'resp_z_thr', 3, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'assess_rejected', true, @islogical);
    addParameter(p, 'do_sorting_metrics', true, @islogical);
    addParameter(p, 'do_rasters', true, @islogical);
    addParameter(p, 'do_clu_plots', true, @islogical);   % SPC temperature diagram per recluster arm
    addParameter(p, 'order_by_rank', false, @islogical);
    addParameter(p, 'run_features', true, @islogical);
    addParameter(p, 'do_plots', true, @islogical);
    addParameter(p, 'out_dir', '', @ischar);
    parse(p, varargin{:});
    o = p.Results;

    valid_arms = {'nomask_recluster','masked_recluster','masked_recluster_rescue'};
    o.arms = lower(o.arms(:)');
    bad = setdiff(o.arms, valid_arms);
    assert(isempty(bad), 'validate_recluster_vs_handselect: bad arms: %s', strjoin(bad, ', '));
    assert(~isempty(o.arms), 'validate_recluster_vs_handselect: arms is empty.');
    want = @(nm) ismember(nm, o.arms);
    do_mask   = want('masked_recluster') || want('masked_recluster_rescue');
    do_rescue = want('masked_recluster_rescue');

    assert(isnumeric(channels) && ~isempty(channels), ...
           'validate_recluster_vs_handselect: channels must be a numeric vector.');
    channels = double(channels(:)');
    assert(~isempty(o.handselect_dir), 'validate_recluster_vs_handselect: ''handselect_dir'' is required.');

    repo = fileparts(fileparts(mfilename('fullpath')));   % .../MATLAB_pipeline
    addpath(genpath(repo));

    root = resolve_session_root();
    home = pwd;
    restore_cwd = onCleanup(@() cd(home)); 
    cd(root);

    hs_dir = o.handselect_dir;
    if ~isfolder(hs_dir), hs_dir = fullfile(root, o.handselect_dir); end
    assert(isfolder(hs_dir), 'handselect_dir not found: %s', o.handselect_dir);

    % Unique run tag (second resolution + collision suffix) so back-to-back
    % runs never share out_dir / spikes_recval_* / times_recval_* - you can
    % just re-run with a different stim set / gt_classes, no filename care.
    ts0 = char(datetime('now', 'Format', 'yyyyMMdd_HHmmSS'));
    ts = ts0; k = 0;
    while exist(fullfile(root, ['RECLUSTER_VALIDATION_' ts]), 'dir') || ...
          exist(fullfile(root, ['spikes_recval_' ts]), 'dir')
        k = k + 1; ts = sprintf('%s_%d', ts0, k);
    end
    if isempty(o.out_dir), o.out_dir = fullfile(root, ['RECLUSTER_VALIDATION_' ts]); end
    if ~exist(o.out_dir, 'dir'), mkdir(o.out_dir); end
    cmp_dir = fullfile(o.out_dir, 'comparisons');
    if ~exist(cmp_dir, 'dir'), mkdir(cmp_dir); end

    grapes_stem = grapes_stem_for(o.use_blanks, o.circshiftblanks);
    gtag = matlab.lang.makeValidName(grapes_stem);
    rtag = [gtag '_rescue'];
    fprintf('[recval] root=%s\n[recval] out=%s\n[recval] handselect=%s\n[recval] flavour=%s\n', ...
            root, o.out_dir, hs_dir, grapes_stem);
    fprintf(['[recval] NOTE response/baseline windows come from the "%s" flavour ' ...
             '(blanks: 50-550 / -550..-50 ms). Every arm, incl. the rebuilt ground-truth ' ...
             'raster, is scored on the same windows.\n'], grapes_stem);

    load(fullfile(root, 'NSx.mat'), 'NSx');

    % ---- sanity: requested channels must exist in THIS session + hand-select ----
    nsx_ids = cell2mat({NSx.chan_ID});
    missing_ch = setdiff(channels, nsx_ids);
    assert(isempty(missing_ch), ...
        ['validate_recluster_vs_handselect: channel(s) %s are not in %s/NSx.mat ' ...
         '(this session has %s). You are probably running from the wrong session folder.'], ...
        mat2str(missing_ch), root, mat2str(nsx_ids(:)'));
    % ---- the hand-select GROUND TRUTH is a grapes*.mat in hs_dir ----
    % (the responses are what matter; the clustering used for matching is
    % RECONSTRUCTED from this raster - see reconstruct_cc_from_grapes)
    gd = dir(fullfile(hs_dir, 'grapes*.mat'));
    gd = gd(~startsWith({gd.name}, 'grapes_response_metrics') & ...
            ~endsWith({gd.name}, '_rescue.mat'));
    if ~isempty(o.handselect_grapes)
        gt_grapes_name = o.handselect_grapes;
        assert(exist(fullfile(hs_dir, gt_grapes_name), 'file') == 2, ...
            'handselect_grapes ''%s'' not found in %s', gt_grapes_name, hs_dir);
    else
        assert(~isempty(gd), ...
            ['validate_recluster_vs_handselect: no grapes*.mat in the hand-select folder\n  (%s)\n' ...
             'That raster is the ground truth. Put one there, or pass ''handselect_grapes''.'], hs_dir);
        assert(isscalar(gd), ...
            ['validate_recluster_vs_handselect: %d grapes*.mat files in %s (%s). Keep only the ' ...
             'one you want as ground truth, or pass ''handselect_grapes''.'], ...
            numel(gd), hs_dir, strjoin({gd.name}, ', '));
        gt_grapes_name = gd(1).name;
    end
    gt_is_blanks = contains(lower(gt_grapes_name), 'blank');
    [~, gt_grapes_base] = fileparts(gt_grapes_name);
    gt_flavour = matlab.lang.makeValidName(gt_grapes_base);
    fprintf('[recval] hand-select ground truth: %s  (blanks windows: %d)\n', gt_grapes_name, gt_is_blanks);

    % ---- resolve the stimulus selection (ONE screening block, or explicit) ----
    ep = load(fullfile(root, 'experiment_properties_online3.mat'), 'experiment', 'scr_config_cell');
    n_scr = numel(ep.scr_config_cell);
    if ~isempty(o.scr)
        assert(isscalar(o.scr) && o.scr >= 1 && o.scr <= n_scr, ...
               'validate_recluster_vs_handselect: scr=%g out of range 1..%d', o.scr, n_scr);
        o.stim_list = unique(double(ep.scr_config_cell{o.scr}.pics2use(:)'), 'stable');
        fprintf('[recval] stimulus selection: screening block %d/%d -> %d stimuli (block order kept)\n', ...
                o.scr, n_scr, numel(o.stim_list));
    elseif ischar(o.stim_list) || isstring(o.stim_list)
        allpics = unique(cell2mat(cellfun(@(x) x.pics2use(:)', ep.scr_config_cell, ...
                                          'UniformOutput', false)));
        fprintf('[recval] stimulus selection: ''all'' -> union of every screening block (%d stimuli). ', ...
                numel(allpics));
        fprintf('Pass ''scr'',K or ''stim_list'',[..] to restrict.\n');
    else
        % keep the caller's order EXACTLY - every plot / heatmap / csv row and
        % the stimulus grid follow this order, not the numeric stim id.
        o.stim_list = unique(double(o.stim_list(:)'), 'stable');
        fprintf('[recval] stimulus selection: explicit list of %d stimuli (your order kept)\n', ...
                numel(o.stim_list));
    end
    if isnumeric(o.stim_list)
        try
            nms = string(ep.experiment.ImageNames.name(o.stim_list));
            fprintf('[recval]   %s\n', strjoin(compose('%d=%s', o.stim_list(:), nms(:)), '  '));
        catch
            fprintf('[recval]   stims: %s\n', mat2str(o.stim_list));
        end
    end
    % same stim_list reaches every arm's characterize_grapes_responses, so the
    % zscore / mu_base / std_base baseline (computed ACROSS this list) is
    % identical across arms and the comparison stays apples-to-apples.

    if o.use_blanks
        fe = load(fullfile(root, 'finalevents.mat'));
        assert(isfield(fe, 'seq_beg_blanks_cell') && ~isempty(fe.seq_beg_blanks_cell), ...
            ['validate_recluster_vs_handselect: use_blanks=true but finalevents.mat has no ' ...
             'seq_beg_blanks_cell. Run extract_blank_on_events_ripple (processing_steps) first, ' ...
             'or call with ''use_blanks'',false.']);
    end

    % ================= working spikes copy =================
    src = o.spikes_src;
    if isempty(src)
        d = dir(fullfile(root, 'spikes*')); d = d([d.isdir]);
        d = d(~startsWith({d.name}, 'spikes_recval_'));
        assert(~isempty(d), 'no spikes* folder under %s; pass ''spikes_src''.', root);
        [~, i] = max([d.datenum]);
        src = fullfile(root, d(i).name);
    elseif ~isfolder(src)
        src = fullfile(root, o.spikes_src);
    end
    assert(isfolder(src), 'spikes_src not found: %s', o.spikes_src);
    spk_work = fullfile(root, ['spikes_recval_' ts]);
    if ~exist(spk_work, 'dir'), mkdir(spk_work); end
    copy_channel_files(src, spk_work, channels, NSx, '_spikes.mat');
    got = arrayfun(@(ch) exist(fullfile(spk_work, [output_name_for(NSx, ch) '_spikes.mat']), 'file') == 2, ...
                   channels);
    assert(any(got), ...
        ['validate_recluster_vs_handselect: none of the requested channels have a ' ...
         '*_spikes.mat in the spikes source\n  (%s)\nRun Get_spikes for %s first, or point ' ...
         '''spikes_src'' at the right folder.'], src, mat2str(channels(:)'));
    if ~all(got)
        warning('recval:someChannelsNoSpikes', 'no spikes for channel(s) %s - they will be skipped.', ...
                mat2str(channels(~got)));
        channels = channels(got);
    end
    fprintf('[recval] working spikes: %s (from %s)\n', spk_work, src);

    d_hs = fullfile(o.out_dir, 'handselect');
    arm_dirs = struct('nomask_recluster', fullfile(o.out_dir, 'nomask_recluster'), ...
                      'masked_recluster', fullfile(o.out_dir, 'masked_recluster'), ...
                      'masked_recluster_rescue', fullfile(o.out_dir, 'masked_recluster_rescue'));

    arm_info = struct('name', {}, 'dir', {}, 'metrics', {}, 'flavour', {}, 'rescue', {});

    % stimulus-onset lookup (snum -> onset times, in update_grapes build order),
    % used to reconstruct an absolute-time cluster_class from any grapes raster
    onset_map = build_onset_map(root);

    % ================= handselect arm =================
    % The ground-truth GRAPES is used AS-IS (never rebuilt) - it is the ONLY
    % thing that has to be right. Its clustering is RECONSTRUCTED from the raster
    % (responses, class matching and assess all flow from that). A REAL times
    % file for that reconstruction is then synthesised from the working spikes
    % file's waveforms (matched by spike time), so cluster-quality metrics and
    % the waveform overlay all run off the grapes clustering too. The folder's
    % own times_*.mat is only used if it happens to BE the same clustering.
    fprintf('\n[recval] === handselect (%s, as-is) ===\n', gt_grapes_name);
    if ~exist(d_hs, 'dir'), mkdir(d_hs); end
    copyfile(fullfile(hs_dir, gt_grapes_name), fullfile(d_hs, gt_grapes_name));
    for ch = channels
        nm = output_name_for(NSx, ch);
        if isempty(nm), continue; end
        gt_cc = reconstruct_cc_from_grapes(fullfile(d_hs, gt_grapes_name), ch, onset_map);
        recon_meta = gt_cc;
        cluster_class = gt_cc.cc;
        save(fullfile(d_hs, ['reconstructed_cc_' nm '.mat']), 'cluster_class', 'recon_meta');
        fprintf('[recval]   ch%d reconstructed cluster_class: %d spikes, classes %s%s\n', ...
                ch, size(gt_cc.cc, 1), mat2str(gt_cc.classes), ...
                iif(gt_cc.ok, '', '  [INCOMPLETE - some stim trial counts did not match]'));

        hs_tf  = fullfile(hs_dir, ['times_' nm '.mat']);
        dst_tf = fullfile(d_hs, ['times_' nm '.mat']);
        if exist(hs_tf, 'file') && times_matches_recon(hs_tf, gt_cc, o.match_tol_ms)
            copy_if(hs_tf, d_hs);
            copy_if(fullfile(hs_dir, [nm '_spikes.mat']), d_hs);
            fprintf('[recval]   ch%d: hand-select times_%s.mat matches the grapes - used as-is.\n', ch, nm);
        else
            W = gt_times_from_grapes(spk_work, nm, gt_cc);
            if ~isempty(W)
                cluster_class = W.cluster_class; spikes = W.spikes; inspk = W.inspk;
                par = W.par; features = W.features;
                save(dst_tf, 'cluster_class', 'spikes', 'inspk', 'par', 'features');
                copy_if(fullfile(spk_work, [nm '_spikes.mat']), d_hs);
                fprintf(['[recval]   ch%d: synthesised times_%s.mat from the grapes reconstruction ' ...
                         '(%d/%d spikes waveform-matched) for cluster quality + overlay.\n'], ...
                        ch, nm, size(W.spikes,1), size(gt_cc.cc,1));
            else
                fprintf('[recval]   ch%d: could not synthesise a GT times file - cluster quality/overlay skipped.\n', ch);
            end
        end
    end
    copy_if(fullfile(hs_dir, 'templates_wc_offline.mat'), d_hs);
    hs_metrics = run_characterize(d_hs, channels, o, root);
    run_sorting_metrics(d_hs, channels, o, root, false);
    if o.do_rasters
        run_rasters_flavour(d_hs, channels, o, root, false, gt_is_blanks);
    end
    arm_info(end+1) = struct('name', 'handselect', 'dir', d_hs, 'metrics', hs_metrics, ...
                             'flavour', gt_flavour, 'rescue', false);

    % ================= nomask_recluster arm =================
    tim_nomask = fullfile(root, ['times_recval_nomask_' ts]);
    if want('nomask_recluster')
        fprintf('\n[recval] === nomask_recluster ===\n');
        recluster(channels, spk_work, tim_nomask, o);
        D = arm_dirs.nomask_recluster;
        snapshot(spk_work, tim_nomask, D, channels, NSx);
        build_grapes(D, channels, o, root, false);
        m = run_characterize(D, channels, o, root);
        run_sorting_metrics(D, channels, o, root, false);
        run_rasters(D, channels, o, root, false);
        arm_info(end+1) = struct('name', 'nomask_recluster', 'dir', D, 'metrics', m, ...
                                 'flavour', gtag, 'rescue', false);
    end

    % ================= masks =================
    if do_mask
        fprintf('\n[recval] within_channel + refract_viol on the working copy\n');
        within_channel(channels, 'detections', o.within_channel_detections);
        refract_viol(channels, 'keep_strategy', o.refract_keep_strategy);
    end

    % ================= masked_recluster arm =================
    tim_masked = fullfile(root, ['times_recval_masked_' ts]);
    if want('masked_recluster') || do_rescue
        fprintf('\n[recval] === masked_recluster ===\n');
        recluster(channels, spk_work, tim_masked, o);
        if want('masked_recluster')
            D = arm_dirs.masked_recluster;
            snapshot(spk_work, tim_masked, D, channels, NSx);
            build_grapes(D, channels, o, root, false);
            m = run_characterize(D, channels, o, root);
            run_sorting_metrics(D, channels, o, root, false);
            run_rasters(D, channels, o, root, false);
            arm_info(end+1) = struct('name', 'masked_recluster', 'dir', D, 'metrics', m, ...
                                     'flavour', gtag, 'rescue', false);
        end
    end

    % ================= masked_recluster_rescue arm =================
    if do_rescue
        fprintf('\n[recval] === masked_recluster_rescue ===\n');
        rescue_spikes(channels, 'parallel', true, 'restore', false, ...
                      'folder_name', ['times_recval_masked_' ts], 'spikes_folder', spk_work);
        D = arm_dirs.masked_recluster_rescue;
        snapshot(spk_work, tim_masked, D, channels, NSx);
        for ch = channels
            patch_rescue_times(fullfile(D, ['times_' output_name_for(NSx, ch) '.mat']));
        end
        build_grapes(D, channels, o, root, true);
        m = run_characterize(D, channels, o, root);
        run_sorting_metrics(D, channels, o, root, true);
        run_rasters(D, channels, o, root, true);
        arm_info(end+1) = struct('name', 'masked_recluster_rescue', 'dir', D, 'metrics', m, ...
                                 'flavour', rtag, 'rescue', true);
    end

    % ---- collect the SPC temperature diagrams into comparisons/ ----
    if o.do_clu_plots
        for ai = 1:numel(arm_info)
            for ch = channels
                nm = output_name_for(NSx, ch);
                src = fullfile(arm_info(ai).dir, ['temperature_' nm '.png']);
                if exist(src, 'file')
                    copyfile(src, fullfile(cmp_dir, sprintf('temperature_ch%d_%s.png', ...
                             ch, arm_info(ai).name)));
                end
            end
        end
    end

    % ================= class matching =================
    % Both sides are reconstructed from their grapes rasters (same onset replay,
    % same post-onset coverage) so recall / precision are apples-to-apples. The
    % ground-truth grapes is the sole source of the hand-select clustering.
    fprintf('\n[recval] class matching (reconstructed rasters, tol=%.2f ms)\n', o.match_tol_ms);
    recluster_arms = arm_info(~strcmp({arm_info.name}, 'handselect'));
    gt_cc_cache = containers.Map('KeyType', 'double', 'ValueType', 'any');
    for ch = channels
        gt_cc_cache(ch) = reconstruct_cc_from_grapes(fullfile(d_hs, gt_grapes_name), ch, onset_map);
    end
    match_tbl = table(); detail_tbl = table(); comp_tbl = table();
    for a = 1:numel(recluster_arms)
        A = recluster_arms(a);
        arm_grapes = fullfile(A.dir, [erase(A.flavour, '_rescue') '.mat']);
        for ch = channels
            gt_cc = gt_cc_cache(ch);
            if isempty(gt_cc.cc) || ~gt_cc.ok
                warning('recval:reconIncomplete', ...
                    'ch%d: ground-truth reconstruction unusable - skipping matching for arm %s.', ...
                    ch, A.name);
                continue;
            end
            if ~exist(arm_grapes, 'file')
                warning('recval:noArmGrapes', 'ch%d arm %s: %s missing - skipped.', ch, A.name, arm_grapes);
                continue;
            end
            ar_cc = reconstruct_cc_from_grapes(arm_grapes, ch, onset_map);
            fprintf('[recval] %-24s ch%d: ground truth %d classes %s  ->  arm %d classes %s\n', ...
                    A.name, ch, numel(gt_cc.classes), mat2str(gt_cc.classes), ...
                    numel(ar_cc.classes), mat2str(ar_cc.classes));
            if numel(ar_cc.classes) < numel(gt_cc.classes)
                fprintf(['[recval]   NOTE arm has FEWER clusters than the hand sort - fresh SPC ' ...
                         'under-segments; the human split further in the GUI. Matches will read ' ...
                         '"merged".\n']);
            end
            [T, DET, COMP] = match_cc(gt_cc.cc, ar_cc.cc, ch, o.match_tol_ms, o.match_min_recall);
            T.arm    = repmat(string(A.name), height(T), 1);
            DET.arm  = repmat(string(A.name), height(DET), 1);
            COMP.arm = repmat(string(A.name), height(COMP), 1);
            match_tbl  = [match_tbl;  T];
            detail_tbl = [detail_tbl; DET];
            comp_tbl   = [comp_tbl;   COMP];
            for r = 1:height(T)
                fprintf(['[recval]   %-24s ch%d  gt class %d -> arm class %d  ' ...
                         '(recall %.2f prec %.2f  %s; %.0f%% unmatched, %.0f%% elsewhere; ' ...
                         'arm cluster %d spk: %.0f%% this class, %d other-gt, %d new)\n'], ...
                        A.name, ch, T.gt_class(r), T.arm_class(r), T.recall(r), T.precision(r), ...
                        T.status(r), 100*T.frac_unmatched(r), 100*T.frac_elsewhere(r), ...
                        T.n_arm(r), 100*T.precision(r), T.n_arm_from_other_gt(r), T.n_arm_new(r));
            end
        end
    end
    out.match = struct(); out.match_detail = detail_tbl; out.match_composition = comp_tbl;
    if ~isempty(match_tbl)
        match_tbl = movevars(match_tbl, {'arm','channel_id','gt_class'}, 'Before', 1);
        writetable(match_tbl,  fullfile(cmp_dir, 'class_match.csv'));
        save(fullfile(cmp_dir, 'class_match.mat'), 'match_tbl', 'detail_tbl', 'comp_tbl');
        if ~isempty(detail_tbl)
            writetable(movevars(detail_tbl, 'arm', 'Before', 1), ...
                       fullfile(cmp_dir, 'class_match_where_gt_spikes_go.csv'));
        end
        if ~isempty(comp_tbl)
            writetable(movevars(comp_tbl, 'arm', 'Before', 1), ...
                       fullfile(cmp_dir, 'class_match_arm_cluster_composition.csv'));
        end
        for a = 1:numel(recluster_arms)
            nmn = recluster_arms(a).name;
            out.match.(matlab.lang.makeValidName(nmn)) = ...
                match_tbl(match_tbl.arm == string(nmn), :);
        end
        if o.do_plots
            try
                class_match_figure(detail_tbl, comp_tbl, channels, {recluster_arms.name}, cmp_dir);
            catch ME
                warning('recval:classMatchFigure', '%s', ME.message);
            end
        end
    else
        warning('[recval] class matching produced no rows.');
    end

    % ================= response-profile comparison + summary =================
    fprintf('\n[recval] response-profile comparison per (channel, ground-truth class)\n');
    out.compare = struct();
    summary = table();
    for ch = channels
        gts = resolve_gt_classes(hs_metrics, arm_info(1).flavour, ch, o.gt_classes);
        for c = gts
            key = sprintf('ch%d_gtclass%d', ch, c);
            [cmp, srows] = build_compare(ch, c, arm_info, match_tbl, o, cmp_dir);
            if isempty(cmp)
                fprintf('[recval]   %s: <2 usable arm columns - skipped.\n', key);
                continue;
            end
            fprintf('[recval]   %s: %d arm columns, %d stim\n', key, ...
                    numel(cmp.arm_labels), numel(cmp.stim_number));
            out.compare.(matlab.lang.makeValidName(key)) = cmp;
            summary = [summary; srows];
            if o.do_plots
                try
                    compare_plots(cmp, fullfile(cmp_dir, [key '_plots']));
                catch ME
                    warning('[recval] plot %s failed: %s', key, ME.message);
                end
                try
                    waveform_overlay(ch, c, arm_info, match_tbl, NSx, spk_work, gt_cc_cache, ...
                                     fullfile(cmp_dir, [key '_plots']));
                catch ME
                    warning('[recval] waveform overlay %s failed: %s', key, ME.message);
                end
            end
        end
    end
    if ~isempty(summary)
        summary = movevars(summary, {'arm','channel_id','gt_class'}, 'Before', 1);
        writetable(summary, fullfile(cmp_dir, 'summary.csv'));
    end
    out.summary = summary;

    % ================= cluster-quality metrics across arms (matched class) ====
    if o.do_sorting_metrics
        fprintf('\n[recval] stacking cluster-quality metrics across arms (matched class)\n');
        try
            out.sorting_metrics = collect_sorting_metrics_matched( ...
                arm_info, match_tbl, channels, hs_metrics, o, cmp_dir);
        catch ME
            warning('[recval] collect_sorting_metrics_matched failed: %s', getReport(ME));
        end
    end

    % ================= assess rejected / rescued =================
    out.assess = struct();
    if o.assess_rejected
        fprintf('\n[recval] assess rejected / rescued sub-populations\n');
        try
            out.assess = assess_reject_rescue(channels, NSx, d_hs, spk_work, ...
                                              arm_info, match_tbl, gtag, rtag, gt_grapes_name, ...
                                              onset_map, o, root);
        catch ME
            warning('[recval] assess_reject_rescue failed: %s', getReport(ME));
        end
    end

    % ================= cluster make-up comparison =========================
    if o.do_plots
        fprintf('\n[recval] cluster make-up comparison (correlograms / waveform sim / spike overlap)\n');
        try
            cluster_comparison(arm_info, channels, NSx, gt_cc_cache, cmp_dir);
        catch ME
            warning('[recval] cluster_comparison failed: %s', getReport(ME));
        end
        try
            out.feature_space = feature_space_plot(arm_info, channels, NSx, hs_dir, gt_cc_cache, cmp_dir);
        catch ME
            warning('[recval] feature_space_plot failed: %s', getReport(ME));
        end
    end

    % ================= per-key dashboards + big-scale HTML report ===========
    if o.do_plots
        fprintf('\n[recval] per-key dashboards\n');
        for kk = fieldnames(out.compare)'
            try
                per_key_dashboard(out.compare.(kk{1}), out, cmp_dir);
            catch ME
                warning('[recval] dashboard %s failed: %s', kk{1}, ME.message);
            end
        end
        try
            rp = write_report(out, arm_info, o, cmp_dir);
            fprintf('[recval] report: %s\n', rp);
        catch ME
            warning('[recval] write_report failed: %s', getReport(ME));
        end
    end

    out.out_dir = o.out_dir;
    out.arms = arm_info;
    out.params = o;
    fprintf('\n[recval] DONE. Outputs in %s\n', o.out_dir);
end

% ======================================================================
% arm building
% ======================================================================
function recluster(channels, spk_work, tim_dir, o)
    if ~exist(tim_dir, 'dir'), mkdir(tim_dir); end
    if o.run_features
        fprintf('[recval] Do_features\n');
        Do_features(channels, 'parallel', true, 'folder', spk_work);
    end
    % byte-identical to processing_steps_MCW_reduced's clustering param, unless
    % the caller overrides via 'clu_par' (e.g. struct('maxtemp',0.32) to let SPC
    % split more, matching what a GUI operator does by hand).
    clu_par = struct('min_clus', 20, 'max_spk', 30000, 'mintemp', 0, 'maxtemp', 0.251, ...
                     'tempstep', 0.01, 'max_std_templates', 3, 'max_spikes_plot', 5000);
    for f = fieldnames(o.clu_par)', clu_par.(f{1}) = o.clu_par.(f{1}); end
    fprintf('[recval] Do_clustering (make_times, make_templates, seed=%g, maxtemp=%g, min_clus=%g)\n', ...
            o.seed, clu_par.maxtemp, clu_par.min_clus);
    Do_clustering(channels, 'parallel', true, 'make_times', true, 'make_templates', true, ...
                  'make_plots', false, 'par', clu_par, 'seed', o.seed, ...
                  'folder', spk_work, 'times_folder', tim_dir);
    % build a clean stand-alone temperature diagram from the SPC tree
    % (data_<nm>.dg_01) - not the full wave_clus 3x5 figure
    if o.do_clu_plots
        s = load(fullfile(resolve_session_root(), 'NSx.mat'), 'NSx');
        for ch = channels
            nm = output_name_for(s.NSx, ch);
            try
                temperature_diagram(tim_dir, nm, clu_par, ...
                                    fullfile(tim_dir, ['temperature_' nm '.png']));
            catch ME
                warning('[recval] temperature_diagram %s: %s', nm, ME.message);
            end
        end
    end
end

function build_grapes(D, channels, o, root, do_rescue)
    cd(root);
    do_structure_mu_BCM_online3(channels, o.exp_type, o.use_blanks, o.circshiftblanks, ...
                                false, 'spike_dir', D);
    do_structure_sorted_BCM_online3(channels, o.use_blanks, o.circshiftblanks, false, ...
                                    'times_dir', D, 'spike_dir', D);
    if do_rescue
        do_structure_rescue_BCM_online3(channels, o.use_blanks, o.circshiftblanks, false, ...
                                        'times_dir', D, 'spike_dir', D);
    end
end

function mpath = run_characterize(D, channels, o, root)
    run_in(D, @() characterize_grapes_responses('channels2plot', channels, ...
        'stim_list', o.stim_list, 'session_root', root));
    d = dir(fullfile(D, 'grapes_response_metrics_*.mat'));
    if isempty(d)
        error('run_characterize: no grapes_response_metrics_*.mat written in %s', D);
    end
    [~, i] = max([d.datenum]);
    mpath = fullfile(d(i).folder, d(i).name);
end

function run_sorting_metrics(D, channels, o, root, is_rescue)
    if ~o.do_sorting_metrics, return; end
    cd(root);
    try
        compute_metrics_batch(channels, 'times_dir', D, 'spikes_dir', D, ...
                              'save', true, 'parallel', true, 'rescue', is_rescue);
    catch ME
        warning('[recval] compute_metrics_batch failed in %s: %s', D, ME.message);
    end
end

function run_rasters(D, channels, o, root, is_rescue)
    run_rasters_flavour(D, channels, o, root, is_rescue, o.use_blanks);
end

function run_rasters_flavour(D, channels, o, root, is_rescue, use_blanks)
% plot_grapes_as_online into the arm folder. order_by_rank stays as configured
% (default false) so stimulus N sits on the same raster row in every arm.
% use_blanks is passed explicitly so the handselect arm can render its own
% flavour (which may be plain 'grapes') rather than o.use_blanks.
    if ~o.do_rasters, return; end
    cd(root);
    args = {'grapes_offline', true, 'target_dir', D, 'channels2plot', channels, ...
            'use_blanks', use_blanks, 'circshiftblanks', o.circshiftblanks, ...
            'stim_list', o.stim_list, 'order_by_rank', o.order_by_rank, ...
            'plot_best_stims_only', false, 'copy2miniscrfolder', false, 'is_online', false};
    if is_rescue
        args = [args, {'rescue', true, 'rescue_units', 'post'}];
    end
    try
        plot_grapes_as_online(args{:});
    catch ME
        warning('[recval] plot_grapes_as_online failed in %s: %s', D, ME.message);
    end
end

% ======================================================================
% cluster-quality metrics stacked across arms, keyed by matched class
% ======================================================================
function ST = collect_sorting_metrics_matched(arm_info, match_tbl, channels, hs_metrics, o, cmp_dir)
    ST = table();
    gt_flav = arm_info(1).flavour;
    for ai = 1:numel(arm_info)
        A = arm_info(ai);
        mdir = fullfile(A.dir, 'metrics');
        if A.rescue && isfolder(fullfile(A.dir, 'metrics rescue'))
            mdir = fullfile(A.dir, 'metrics rescue');
        end
        d = dir(fullfile(mdir, '*_metrics.mat'));
        armq = table();
        for k = 1:numel(d)
            S = load(fullfile(d(k).folder, d(k).name));
            if ~isfield(S, 'metrics_table') || ~istable(S.metrics_table) || isempty(S.metrics_table)
                continue;
            end
            armq = [armq; S.metrics_table];
        end
        if isempty(armq) || ~ismember('channel_id', armq.Properties.VariableNames), continue; end

        for ch = channels
            gts = resolve_gt_classes(hs_metrics, gt_flav, ch, o.gt_classes);
            for c = gts
                if strcmp(A.name, 'handselect')
                    arm_cls = c; status = "matched";
                else
                    mr = match_tbl(match_tbl.arm == string(A.name) & ...
                                   match_tbl.channel_id == ch & match_tbl.gt_class == c, :);
                    if isempty(mr), continue; end
                    arm_cls = mr.arm_class(1); status = mr.status(1);
                end
                qr = armq(armq.channel_id == ch & armq.cluster_id == arm_cls, :);
                if isempty(qr), continue; end
                lead = table(string(A.name), ch, c, arm_cls, string(status), ...
                    'VariableNames', {'arm','channel_id','gt_class','arm_class','match_status'});
                qr = removevars(qr, intersect({'channel_id','cluster_id','filename'}, ...
                                              qr.Properties.VariableNames));
                ST = [ST; [lead, qr(1, :)]];
            end
        end
    end
    if isempty(ST), return; end
    writetable(ST, fullfile(cmp_dir, 'sorting_metrics.csv'));
    save(fullfile(cmp_dir, 'sorting_metrics.mat'), 'ST');

    if ~o.do_plots, return; end
    metric_cols = ST.Properties.VariableNames(varfun(@isnumeric, ST, 'OutputFormat', 'uniform'));
    metric_cols = setdiff(metric_cols, {'channel_id','gt_class','arm_class'}, 'stable');
    arms = unique(ST.arm, 'stable');
    for ch = channels
        for c = unique(ST.gt_class(ST.channel_id == ch))'
            sub = ST(ST.channel_id == ch & ST.gt_class == c, :);
            if height(sub) < 2, continue; end
            f = figure('Visible', 'off', 'Position', [100 100 max(1000, 240*numel(arms)) 760], 'Color', 'w');
            tl = tiledlayout(f, 'flow', 'Padding', 'loose', 'TileSpacing', 'loose');
            title(tl, sprintf('ch%d  ground-truth class %d  cluster quality across arms', ch, c), ...
                  'Interpreter', 'none');
            [~, ia] = ismember(sub.arm, arms); x = ia;
            for mm = 1:numel(metric_cols)
                ax = nexttile(tl);
                y = double(sub.(metric_cols{mm}));
                plot(ax, x, y, '-o', 'LineWidth', 1.2); grid(ax, 'on');
                xlim(ax, [0.5 numel(arms)+0.5]); xticks(ax, 1:numel(arms));
                xticklabels(ax, arms); set(ax, 'TickLabelInterpreter', 'none');
                title(ax, metric_cols{mm}, 'Interpreter', 'none');
            end
            save_fig(f, fullfile(cmp_dir, sprintf('sorting_metrics_ch%d_gtclass%d.png', ch, c)));
        end
    end
end

% ======================================================================
% GT-vs-arm mean waveform overlay
% ======================================================================
function waveform_overlay(ch, gt_class, arm_info, match_tbl, NSx, spk_work, gt_cc_cache, plot_dir)
    if ~exist(plot_dir, 'dir'), mkdir(plot_dir); end
    nm = output_name_for(NSx, ch);
    f = figure('Visible', 'off', 'Position', [100 100 900 520], 'Color', 'w');
    ax = axes(f); hold(ax, 'on');
    cols = lines(numel(arm_info) + 1);
    leg = {}; hh = []; plotted = false;
    for ai = 1:numel(arm_info)
        A = arm_info(ai);
        is_hs = strcmp(A.name, 'handselect');
        if is_hs
            % grapes has no waveforms - pull them from the working spikes file
            % by matching the reconstructed GT-class spike times to index_all
            gt_cc = []; if isKey(gt_cc_cache, ch), gt_cc = gt_cc_cache(ch); end
            w = gt_waveforms(spk_work, nm, gt_cc, gt_class);
            acl = gt_class;
        else
            tf = fullfile(A.dir, ['times_' nm '.mat']);
            if ~exist(tf, 'file'), continue; end
            S = load(tf, 'spikes', 'cluster_class');
            if ~isfield(S, 'spikes') || ~isfield(S, 'cluster_class'), continue; end
            mr = match_tbl(match_tbl.arm == string(A.name) & match_tbl.channel_id == ch & ...
                           match_tbl.gt_class == gt_class, :);
            if isempty(mr) || mr.n_shared(1) == 0, continue; end
            acl = mr.arm_class(1);
            w = S.spikes(S.cluster_class(:, 1) == acl, :);
        end
        if isempty(w), continue; end
        mu = mean(w, 1); sd = std(w, 0, 1); xx = 1:numel(mu);
        lw = 2.5 * is_hs + 1.2 * ~is_hs;
        cc = cols(ai, :); if is_hs, cc = [0 0 0]; end
        fill(ax, [xx fliplr(xx)], [mu+sd fliplr(mu-sd)], cc, ...
             'FaceAlpha', 0.12, 'EdgeColor', 'none', 'HandleVisibility', 'off');
        h = plot(ax, xx, mu, '-', 'Color', cc, 'LineWidth', lw);
        hh(end+1) = h;
        if is_hs
            leg{end+1} = sprintf('handselect (gt c%d, n=%d)', gt_class, size(w, 1));
        else
            leg{end+1} = sprintf('%s (c%d, n=%d)', A.name, acl, size(w, 1));
        end
        plotted = true;
    end
    if ~plotted, close(f); return; end
    hold(ax, 'off'); grid(ax, 'on');
    legend(ax, hh, leg, 'Interpreter', 'none', 'Location', 'best');
    title(ax, sprintf('ch%d  ground-truth class %d  mean waveform +/- std', ch, gt_class), ...
          'Interpreter', 'none');
    xlabel(ax, 'sample'); ylabel(ax, 'amplitude');
    save_fig(f, fullfile(plot_dir, 'waveform_overlay.png'));
end

function w = gt_waveforms(spk_work, nm, gt_cc, gt_class)
% Mean-waveform material for a ground-truth class: match its reconstructed
% spike times to the working spikes file's index_all and take those waveforms.
    w = [];
    sf = fullfile(spk_work, [nm '_spikes.mat']);
    if ~exist(sf, 'file') || isempty(gt_cc) || isempty(gt_cc.cc), return; end
    S = load(sf);
    [SA, IA] = spk_all(S);
    if isempty(SA), return; end
    gt_t = gt_cc.cc(gt_cc.cc(:,1) == gt_class, 2);
    if isempty(gt_t), return; end
    idx = nearest_match(gt_t, IA, 0.5);
    idx = idx(idx > 0);
    if isempty(idx), return; end
    w = SA(idx, :);
end

function W = gt_times_from_grapes(spk_work, nm, gt_cc)
% Synthesise a REAL times_<nm>.mat for the grapes reconstruction: match every
% reconstructed spike to the working spikes file by time, take that waveform,
% compute wavelet features for exactly this subset. -> cluster_class / spikes /
% inspk / par, so compute_metrics_batch + waveform_overlay run off the grapes.
    W = [];
    sf = fullfile(spk_work, [nm '_spikes.mat']);
    if ~exist(sf, 'file') || isempty(gt_cc) || isempty(gt_cc.cc), return; end
    S = load(sf);
    [SA, IA] = spk_all(S);
    if isempty(SA) || ~isfield(S, 'par'), return; end
    idx = nearest_match(gt_cc.cc(:,2), IA, 0.5);
    keep = idx > 0;
    if nnz(keep) < 20, return; end
    spikes = SA(idx(keep), :);
    cc = [gt_cc.cc(keep, 1), gt_cc.cc(keep, 2)];
    try
        feats = wave_features(spikes, S.par);
        inspk = feats.inspk;
    catch ME
        warning('recval:gtFeatures', 'wave_features failed for %s: %s', nm, ME.message);
        return;
    end
    W = struct('cluster_class', cc, 'spikes', spikes, 'inspk', inspk, 'par', S.par, 'features', feats);
end

function t = recon_all_spike_times(grapes_file, ch, omap)
% Absolute times of EVERY spike in the mu raster (post-onset), de-duplicated.
    t = [];
    try
        G = load(grapes_file);
    catch
        return;
    end
    cf = ['chan' num2str(ch)];
    if ~isfield(G, 'rasters') || ~isfield(G.rasters, cf) || ~isfield(G.rasters.(cf), 'mu')
        return;
    end
    stimc = G.rasters.(cf).mu.stim;
    allt = [];
    for snum = 1:numel(stimc)
        tr = stimc{snum};
        if isempty(tr) || ~isKey(omap, snum), continue; end
        if ~iscell(tr), tr = num2cell(tr, 2); end
        O = omap(snum);
        if numel(O) ~= numel(tr), continue; end
        for k = 1:numel(tr)
            rel = tr{k}(:).'; rel = rel(rel >= 0 & rel < 9000);
            if ~isempty(rel), allt = [allt, rel + O(k)]; end
        end
    end
    allt = sort(allt);
    if ~isempty(allt), allt = allt([true, diff(allt) > 0.05]); end
    t = allt(:);
end

function [SA, IA] = spk_all(S)
    SA = []; IA = [];
    if isfield(S, 'spikes_all') && isfield(S, 'index_all')
        SA = S.spikes_all; IA = double(S.index_all(:));
    elseif isfield(S, 'spikes') && isfield(S, 'index')
        SA = S.spikes; IA = double(S.index(:));
    end
end

% ======================================================================
% class matching -- match_cc moved to responses/match_cc.m (shared with
% analyze_cluster_membership.m).
% ======================================================================

% ======================================================================
% response comparison
% ======================================================================
function classes = resolve_gt_classes(metrics_file, gtag, ch, gt_classes)
    classes = [];
    m = load(metrics_file);
    if ~isfield(m, gtag), return; end
    Tt = m.(gtag).metrics;
    sel = Tt.channel_num == ch;
    cl = unique(cellstr(string(Tt.class(sel))));
    for i = 1:numel(cl)
        tok = regexp(cl{i}, '^class(\d+)$', 'tokens', 'once');
        if ~isempty(tok), classes(end+1) = str2double(tok{1}); end
    end
    classes = sort(unique(classes));
    if ~(ischar(gt_classes) || isstring(gt_classes))
        classes = intersect(classes, double(gt_classes(:)'));
    end
end

function [cmp, srows] = build_compare(ch, gt_class, arm_info, match_tbl, o, cmp_dir)
% stim x arm matrices of every response metric for ground-truth class gt_class,
% each arm read at its matched class. Column 1 is always handselect (read at its
% own grapes flavour, which may differ from the recluster arms').
    cmp = []; srows = table();

    % ---- assemble columns ----
    z = {'label',{},'file',{},'tag',{},'class',{},'status',{}, ...
         'recall',{},'precision',{},'frac_unmatched',{},'frac_elsewhere',{}, ...
         'n_arm',{},'n_arm_new',{}};
    cols = struct(z{:});
    cols(end+1) = struct('label','handselect','file',arm_info(1).metrics,'tag',arm_info(1).flavour, ...
        'class',sprintf('class%d',gt_class),'status',"matched", ...
        'recall',1,'precision',1,'frac_unmatched',0,'frac_elsewhere',0,'n_arm',NaN,'n_arm_new',0);
    recluster_arms = arm_info(~strcmp({arm_info.name}, 'handselect'));
    for a = 1:numel(recluster_arms)
        A = recluster_arms(a);
        mr = match_tbl(match_tbl.arm == string(A.name) & match_tbl.channel_id == ch & ...
                       match_tbl.gt_class == gt_class, :);
        if isempty(mr), continue; end
        if mr.recall(1) < 0.2 && mr.n_shared(1) == 0
            fprintf('[recval]   ch%d gtclass%d: arm %s has no overlapping class - dropped.\n', ...
                    ch, gt_class, A.name);
            continue;
        end
        cols(end+1) = struct('label', sprintf('%s(c%d,%s)', A.name, mr.arm_class(1), mr.status(1)), ...
                             'file', A.metrics, 'tag', A.flavour, ...
                             'class', sprintf('class%d', mr.arm_class(1)), ...
                             'status', mr.status(1), 'recall', mr.recall(1), ...
                             'precision', mr.precision(1), 'frac_unmatched', mr.frac_unmatched(1), ...
                             'frac_elsewhere', mr.frac_elsewhere(1), 'n_arm', mr.n_arm(1), ...
                             'n_arm_new', mr.n_arm_new(1));
    end
    if numel(cols) < 2, return; end

    % ---- rows for each column ----
    rows = cell(numel(cols), 1);
    for k = 1:numel(cols)
        S = load(cols(k).file);
        if ~isfield(S, cols(k).tag)
            warning('[recval] %s has no "%s" block - column dropped.', cols(k).file, cols(k).tag);
            rows{k} = table(); continue;
        end
        Tt = S.(cols(k).tag).metrics;
        sel = Tt.channel_num == ch & strcmp(string(Tt.class), cols(k).class);
        rows{k} = Tt(sel, :);
    end
    good = ~cellfun(@isempty, rows);
    cols = cols(good); rows = rows(good);
    if numel(cols) < 2, return; end

    % ---- stimulus set (row order) ----
    % Explicit list  -> EXACTLY the caller's order (already de-duped upstream).
    % 'all'          -> union across arms, in first-seen order (not sorted).
    if ischar(o.stim_list) || isstring(o.stim_list)
        stim_number = [];
        for k = 1:numel(rows)
            stim_number = [stim_number; setdiff(rows{k}.stim_number(:), stim_number, 'stable')];
        end
    else
        stim_number = double(o.stim_list(:));
    end
    stim_number = stim_number(:);
    nstim = numel(stim_number);
    narm  = numel(cols);

    stim_name = strings(nstim, 1);
    for k = 1:numel(rows)
        [tf, loc] = ismember(stim_number, rows{k}.stim_number);
        nm = strings(nstim, 1); nm(tf) = string(rows{k}.stim_name(loc(tf)));
        fill = stim_name == "" & nm ~= ""; stim_name(fill) = nm(fill);
    end

    ex = rows{1};
    id_cols = {'channel','channel_num','ch_label','class','stim_number','stim_name','IFR'};
    metric_cols = setdiff(ex.Properties.VariableNames, id_cols, 'stable');
    keep = varfun(@(v) isnumeric(v) || islogical(v), ex(:, metric_cols), 'OutputFormat', 'uniform');
    metric_cols = metric_cols(keep);

    wide = struct(); delta = struct();
    for mi = 1:numel(metric_cols)
        M = nan(nstim, narm);
        for k = 1:narm
            [tf, loc] = ismember(stim_number, rows{k}.stim_number);
            v = double(rows{k}.(metric_cols{mi}));
            M(tf, k) = v(loc(tf));
        end
        wide.(metric_cols{mi})  = M;
        delta.(metric_cols{mi}) = M - M(:, 1);
    end

    % ---- responsive sets + per-arm summary ----
    resp = false(nstim, narm);
    for k = 1:narm
        [tf, loc] = ismember(stim_number, rows{k}.stim_number);
        r = false(nstim, 1);
        R = rows{k};
        gate = R.min_spk_test & R.p_test & R.good_lat & (abs(R.zscore) >= o.resp_z_thr);
        r(tf) = gate(loc(tf));
        resp(:, k) = r;
    end
    R0 = resp(:, 1);
    labels = string({cols.label});
    for k = 1:narm
        Rk = resp(:, k);
        inter = nnz(R0 & Rk); uni = nnz(R0 | Rk);
        srows = [srows; table( ...
            string(regexprep(cols(k).label, '\(.*$', '')), ch, gt_class, ...
            string(cols(k).class), string(cols(k).status), ...
            nnz(Rk), mean(Rk), nnz(Rk & ~R0), nnz(R0 & ~Rk), ...
            (inter / max(uni, 1)), mean(wide.zscore(:, k), 'omitnan'), ...
            'VariableNames', {'arm','channel_id','gt_class','arm_class','match_status', ...
                'n_responsive','frac_responsive','n_gained','n_lost','jaccard_resp','mean_zscore'})];
    end

    cmp = struct('channel', ch, 'gt_class', gt_class, 'arm_labels', {cellstr(labels)}, ...
                 'arm_class', {string({cols.class})}, 'arm_status', {string({cols.status})}, ...
                 'match_recall', [cols.recall], 'match_precision', [cols.precision], ...
                 'match_frac_unmatched', [cols.frac_unmatched], ...
                 'match_frac_elsewhere', [cols.frac_elsewhere], ...
                 'match_n_arm', [cols.n_arm], 'match_n_arm_new', [cols.n_arm_new], ...
                 'stim_number', stim_number, 'stim_name', stim_name, ...
                 'metrics_list', {metric_cols}, 'wide', wide, 'delta', delta, ...
                 'responsive', resp, 'resp_z_thr', o.resp_z_thr);
    save(fullfile(cmp_dir, sprintf('compare_ch%d_gtclass%d.mat', ch, gt_class)), '-struct', 'cmp', '-v7');
end

% ======================================================================
% assess rejected / rescued
% ======================================================================
function A = assess_reject_rescue(channels, NSx, d_hs, spk_work, arm_info, ...
                                  match_tbl, gtag, rtag, gt_grapes_name, onset_map, o, root)
% For each ground-truth class: split its (grapes-reconstructed) spikes by mask
% status of the masked working spikes file, build a single-class raster for the
% rejected subsets, and score the stimulus response. The RESCUED population is
% read straight from the rescue arm's classN_rescue block.
    A = struct('rejected', table(), 'rescued', table());
    assess_root = fullfile(o.out_dir, 'assess');
    if ~exist(assess_root, 'dir'), mkdir(assess_root); end
    rescue_arm = arm_info(strcmp({arm_info.name}, 'masked_recluster_rescue'));

    for ch = channels
        nm = output_name_for(NSx, ch);
        spk_f = fullfile(spk_work, [nm '_spikes.mat']);
        if ~exist(spk_f, 'file'), continue; end
        gt_cc = reconstruct_cc_from_grapes(fullfile(d_hs, gt_grapes_name), ch, onset_map);
        cc = gt_cc.cc;
        if isempty(cc), continue; end
        Sp = load(spk_f, 'index_all', 'index', 'mask_non_quarantine', 'mask_non_refract');
        % within_channel/refract_viol write index_all + the masks; if neither
        % ran (no masked arm) there is nothing to assess - fall back to index
        % and skip. get_mask() already defaults a missing mask to all-kept.
        if isfield(Sp, 'index_all') && ~isempty(Sp.index_all)
            idx_all = Sp.index_all(:);
        elseif isfield(Sp, 'index') && ~isempty(Sp.index)
            idx_all = Sp.index(:);
        else
            continue;
        end
        if ~isfield(Sp, 'mask_non_quarantine') && ~isfield(Sp, 'mask_non_refract')
            fprintf('[recval]   assess: ch%d has no artifact masks (no masked arm) - skipped.\n', ch);
            continue;
        end
        mq = get_mask(Sp, 'mask_non_quarantine', numel(idx_all));   % true = kept
        mrf = get_mask(Sp, 'mask_non_refract', numel(idx_all));

        gts = resolve_gt_classes(arm_info(1).metrics, arm_info(1).flavour, ch, o.gt_classes);
        for c = gts
            gt_rows = cc(cc(:,1) == c, :);
            if isempty(gt_rows), continue; end
            gm = nearest_match(gt_rows(:,2), idx_all, o.match_tol_ms);   % index into idx_all or 0
            has = gm > 0;
            st_q = has & ~mq(max(gm, 1));            % rejected by quarantine
            st_r = has & mq(max(gm, 1)) & ~mrf(max(gm, 1));   % refract-only rejection
            subsets = struct('name', {'rejected_quarantine','rejected_refract'}, ...
                             'sel',  {st_q, st_r});
            for s = 1:numel(subsets)
                sub_t = gt_rows(subsets(s).sel, 2);
                row = table(ch, c, string(subsets(s).name), numel(sub_t), ...
                    'VariableNames', {'channel_id','gt_class','subset','n_spikes'});
                if numel(sub_t) >= 5
                    sd = fullfile(assess_root, sprintf('ch%d_gtclass%d_%s', ch, c, subsets(s).name));
                    try
                        mtab = score_subraster(sd, ch, nm, sub_t, o, root, gtag, d_hs);
                        row = [row, summarize_subraster(mtab, o.resp_z_thr)];
                    catch ME
                        warning('[recval] assess subraster ch%d gtclass%d %s failed: %s', ...
                                ch, c, subsets(s).name, ME.message);
                        row = [row, empty_subsummary()];
                    end
                else
                    row = [row, empty_subsummary()];
                end
                A.rejected = [A.rejected; row];
            end
        end
    end

    % ---- rescued population: read classN_rescue from the rescue arm ----
    if ~isempty(rescue_arm)
        Sm = load(rescue_arm.metrics);
        if isfield(Sm, rtag)
            Tt = Sm.(rtag).metrics;
            for ch = channels
                mr = match_tbl(match_tbl.arm == "masked_recluster_rescue" & ...
                               match_tbl.channel_id == ch, :);
                for i = 1:height(mr)
                    rc = sprintf('class%d_rescue', mr.arm_class(i));
                    sel = Tt.channel_num == ch & strcmp(string(Tt.class), rc);
                    if ~any(sel), continue; end
                    sub = summarize_metrics_rows(Tt(sel, :), o.resp_z_thr);
                    A.rescued = [A.rescued; [table(ch, mr.gt_class(i), string(rc), ...
                        'VariableNames', {'channel_id','gt_class','rescue_class'}), sub]];
                end
            end
        end
    end

    ad = fullfile(o.out_dir, 'comparisons');
    if ~isempty(A.rejected), writetable(A.rejected, fullfile(ad, 'assess_rejected.csv')); end
    if ~isempty(A.rescued),  writetable(A.rescued,  fullfile(ad, 'assess_rescued.csv'));  end
end

function mtab = score_subraster(sub_dir, ch, nm, spike_times_ms, o, root, gtag, d_hs)
    if ~exist(sub_dir, 'dir'), mkdir(sub_dir); end
    cluster_class = [ones(numel(spike_times_ms), 1), spike_times_ms(:)];
    save(fullfile(sub_dir, ['times_' nm '.mat']), 'cluster_class');
    strip_stale_grapes(sub_dir);
    % seed a base grapes file: do_structure_sorted only sets time_pre/time_pos
    % when it initialises from scratch, never exp_type, which
    % characterize_grapes_responses then needs. Take the fields from the
    % rebuilt hand-select grapes so the windows match the arms.
    grapes_name = [grapes_stem_for(o.use_blanks, o.circshiftblanks) '.mat'];
    base = struct('exp_type', o.exp_type, 'time_pre', 1e3, 'time_pos', 2e3);
    hg = fullfile(d_hs, grapes_name);
    if exist(hg, 'file')
        G = load(hg);
        for fld = {'exp_type', 'time_pre', 'time_pos', 'ISI_min'}
            if isfield(G, fld{1}) && ~isempty(G.(fld{1})), base.(fld{1}) = G.(fld{1}); end
        end
    end
    save(fullfile(sub_dir, grapes_name), '-struct', 'base');
    cd(root);
    do_structure_sorted_BCM_online3(ch, o.use_blanks, o.circshiftblanks, false, ...
                                    'times_dir', sub_dir, 'spike_dir', sub_dir);
    mp = run_characterize(sub_dir, ch, o, root);
    S = load(mp);
    mtab = table();
    if isfield(S, gtag), mtab = S.(gtag).metrics; end
end

function s = summarize_subraster(mtab, zthr)
    if isempty(mtab), s = empty_subsummary(); return; end
    s = summarize_metrics_rows(mtab, zthr);
end

function s = summarize_metrics_rows(T, zthr)
    gate = T.min_spk_test & T.p_test & T.good_lat & (abs(T.zscore) >= zthr);
    s = table(nnz(gate), mean(gate), mean(T.zscore, 'omitnan'), max(T.zscore, [], 'omitnan'), ...
              mean(T.resp_delta_fr_hz, 'omitnan'), ...
        'VariableNames', {'n_responsive','frac_responsive','mean_zscore','max_zscore','mean_resp_delta_fr_hz'});
end

function s = empty_subsummary()
    s = table(0, 0, NaN, NaN, NaN, ...
        'VariableNames', {'n_responsive','frac_responsive','mean_zscore','max_zscore','mean_resp_delta_fr_hz'});
end

function gm = nearest_match(gt_t, ar_t, tol)
% gm(i) = index into ar_t of the nearest spike to gt_t(i) within tol, else 0.
    gm = zeros(numel(gt_t), 1);
    [as, order] = sort(ar_t(:));
    for i = 1:numel(gt_t)
        k = find(as >= gt_t(i) - tol, 1, 'first');
        best = 0; bestd = tol + 1;
        while ~isempty(k) && k <= numel(as) && as(k) <= gt_t(i) + tol
            dd = abs(as(k) - gt_t(i));
            if dd < bestd, bestd = dd; best = k; end
            k = k + 1;
        end
        if best > 0, gm(i) = order(best); end
    end
end

function m = get_mask(S, field, n)
    if isfield(S, field) && ~isempty(S.(field))
        m = logical(S.(field)(:));
        if numel(m) ~= n, m = true(n, 1); end
    else
        m = true(n, 1);
    end
end

% ======================================================================
% plots
% ======================================================================
function compare_plots(cmp, plot_dir)
    if ~exist(plot_dir, 'dir'), mkdir(plot_dir); end
    labels = shorten(cmp.arm_labels);
    narm = numel(labels);
    stimN = cmp.stim_number(:); stimNm = cmp.stim_name(:);
    nstim = numel(stimN);
    row_lab = arrayfun(@(k) sprintf('%d %s', stimN(k), stimNm(k)), (1:nstim)', 'UniformOutput', false);
    xt = 1:narm;
    metrics = intersect({'zscore','auroc','resp_delta_fr_hz','p_value_sign','peak_ifr', ...
                         'onset','frac_active_trials','first_spk_lat_med'}, cmp.metrics_list, 'stable');
    col_key = strjoin(compose('%d=%s', xt(:), string(labels(:))), '   ');

    % responsive-count bar
    f = figure('Visible', 'off', 'Position', [100 100 max(900, 240*narm) 460], 'Color', 'w');
    ax = axes(f); bar(ax, xt, sum(cmp.responsive, 1));
    xlim(ax, [0.5 narm+0.5]); xticks(ax, xt); xticklabels(ax, labels);
    set(ax, 'TickLabelInterpreter', 'none');
    ylabel(ax, sprintf('# responsive stim (|z|>=%.1f & gate)', cmp.resp_z_thr));
    title(ax, sprintf('ch%d  ground-truth class %d  (%d stim)', cmp.channel, cmp.gt_class, nstim), ...
          'Interpreter', 'none');
    save_fig(f, fullfile(plot_dir, 'responsive_count.png'));

    cmap = lines(max(nstim, 7));
    for mi = 1:numel(metrics)
        m = metrics{mi}; M = cmp.wide.(m);
        if ~any(isfinite(M(:))), continue; end
        f = figure('Visible', 'off', 'Position', [100 100 max(900, 240*narm) 620], 'Color', 'w');
        ax = axes(f); hold(ax, 'on');
        for s = 1:nstim
            plot(ax, xt, M(s, :), '-o', 'Color', cmap(s, :), 'MarkerFaceColor', cmap(s, :));
        end
        hold(ax, 'off'); grid(ax, 'on');
        xlim(ax, [0.5 narm+0.5]); xticks(ax, xt); xticklabels(ax, labels);
        set(ax, 'TickLabelInterpreter', 'none');
        ylabel(ax, m, 'Interpreter', 'none');
        rl = robust_lims(M(:));
        if ~isempty(rl), ylim(ax, rl); end
        title(ax, sprintf('ch%d gtclass%d - %s  (line per stim, x = arm)', ...
              cmp.channel, cmp.gt_class, m), 'Interpreter', 'none');
        save_fig(f, fullfile(plot_dir, sprintf('change_%s.png', m)));

        make_heatmap(M, row_lab, labels, sprintf('ch%d gtclass%d - %s', cmp.channel, cmp.gt_class, m), ...
                     false, fullfile(plot_dir, sprintf('heatmap_%s.png', m)));
        if isfield(cmp.delta, m)
            make_heatmap(cmp.delta.(m), row_lab, labels, ...
                sprintf('ch%d gtclass%d - %s (delta vs handselect)', cmp.channel, cmp.gt_class, m), ...
                true, fullfile(plot_dir, sprintf('heatmap_delta_%s.png', m)));
        end
    end
    fid = fopen(fullfile(plot_dir, 'arm_key.txt'), 'w');
    if fid > 0, fprintf(fid, '%s\n', col_key); fclose(fid); end
end

% ----------------------------------------------------------------------
function per_key_dashboard(cmp, out, cmp_dir)
% One figure per (channel, gt class): grouped bar panels, x = arm, covering
% MATCH quality, RESPONSE outcome and (if available) CLUSTER quality - so the
% sorting change and its effect on the response sit side by side.
    ch = cmp.channel; c = cmp.gt_class;
    labels = shorten(cmp.arm_labels);
    narm = numel(labels); xt = 1:narm;

    sm = out.summary;
    sm = sm(sm.channel_id == ch & sm.gt_class == c, :);
    % align summary rows to cmp.arm order (arm name is label before '(')
    armname = regexprep(cmp.arm_labels, '\(.*$', '');
    getrow = @(nm, col) local_pick(sm, nm, col);

    panels = {};
    panels{end+1} = {'match: recall',        cmp.match_recall};
    panels{end+1} = {'match: precision',     cmp.match_precision};
    panels{end+1} = {'match: frac unmatched',cmp.match_frac_unmatched};
    panels{end+1} = {'match: frac elsewhere',cmp.match_frac_elsewhere};
    panels{end+1} = {'match: arm-cluster new spikes', cmp.match_n_arm_new};
    panels{end+1} = {'resp: # responsive stim', arrayfun(@(k) getrow(armname{k},'n_responsive'), 1:narm)};
    panels{end+1} = {'resp: # gained vs handselect', arrayfun(@(k) getrow(armname{k},'n_gained'), 1:narm)};
    panels{end+1} = {'resp: # lost vs handselect',   arrayfun(@(k) getrow(armname{k},'n_lost'), 1:narm)};
    panels{end+1} = {'resp: jaccard vs handselect',  arrayfun(@(k) getrow(armname{k},'jaccard_resp'), 1:narm)};
    panels{end+1} = {'resp: mean zscore',            arrayfun(@(k) getrow(armname{k},'mean_zscore'), 1:narm)};

    % cluster-quality panels from sorting_metrics (matched class)
    if isfield(out, 'sorting_metrics') && ~isempty(out.sorting_metrics)
        ST = out.sorting_metrics;
        ST = ST(ST.channel_id == ch & ST.gt_class == c, :);
        for q = {'num_spikes','snr','firing_rate','isi_violation_rate','isolation_distance','l_ratio','d_prime'}
            if ~ismember(q{1}, ST.Properties.VariableNames), continue; end
            y = arrayfun(@(k) local_pick(ST, armname{k}, q{1}), 1:narm);
            panels{end+1} = {['qual: ' q{1}], y};
        end
    end

    f = figure('Visible','off','Position',[80 80 max(1200,260*narm) 900],'Color','w');
    tl = tiledlayout(f, 'flow', 'Padding','loose','TileSpacing','loose');
    title(tl, sprintf('ch%d  ground-truth class %d  -  dashboard (x = arm)', ch, c), 'Interpreter','none');
    xlabel(tl, strjoin(compose('%d=%s', xt(:), string(labels(:))), '   '), 'Interpreter','none');
    for pp = 1:numel(panels)
        ax = nexttile(tl);
        y = panels{pp}{2}; y = y(:)';
        bar(ax, xt, y); grid(ax,'on');
        xlim(ax,[0.5 narm+0.5]); xticks(ax, xt); xticklabels(ax, labels);
        set(ax,'TickLabelInterpreter','none');
        title(ax, panels{pp}{1}, 'Interpreter','none');
    end
    save_fig(f, fullfile(cmp_dir, sprintf('dashboard_ch%d_gtclass%d.png', ch, c)));
end

function v = local_pick(T, armname, col)
    v = NaN;
    if isempty(T) || ~ismember(col, T.Properties.VariableNames), return; end
    r = T(strcmp(string(T.arm), string(armname)) | ...
          startsWith(string(T.arm), string(armname)), :);
    if ~isempty(r), v = double(r.(col)(1)); end
end

% ----------------------------------------------------------------------
% class_match_figure moved to responses/class_match_figure.m (shared with
% analyze_cluster_membership.m).

% ======================================================================
% cluster make-up comparison: correlograms, waveform similarity, spike overlap
% ======================================================================
function cluster_comparison(arm_info, channels, NSx, gt_cc_cache, cmp_dir)
% Per channel -> comparisons/cluster_comparison_ch<NN>/ :
%   acg_ccg_<arm>.png       N x N grid: diagonal = auto-correlogram (refractory
%                           dip / bursting), off-diagonal = cross-correlogram
%                           (a dip at lag 0 => the two "clusters" are one unit)
%   waveform_xcorr_<arm>.png   class x class mean-waveform correlation matrix
%                           (high off-diagonal => similar shapes)
%   spike_overlap_<arm>.png    ground-truth class x arm class Jaccard of spike
%                           times (shows the many-GT -> one-arm collapse)
    maxlag = 50; binw = 1;   % ms
    for ci = 1:numel(channels)
        ch = channels(ci);
        nm = output_name_for(NSx, ch);
        od = fullfile(cmp_dir, sprintf('cluster_comparison_ch%d', ch));
        if ~exist(od, 'dir'), mkdir(od); end

        gt = gt_cc_cache(ch);
        if ~isempty(gt.cc)
            corr_grid(gt.cc, sprintf('handselect ch%d  (grapes-reconstructed spikes)', ch), ...
                      fullfile(od, 'acg_ccg_handselect.png'), maxlag, binw);
        end

        for ai = 1:numel(arm_info)
            A = arm_info(ai);
            if strcmp(A.name, 'handselect'), continue; end
            tf = fullfile(A.dir, ['times_' nm '.mat']);
            if ~exist(tf, 'file'), continue; end
            S = load(tf);
            cc = S.cluster_class;
            spk = []; if isfield(S, 'spikes'), spk = S.spikes; end
            corr_grid(cc, sprintf('%s  ch%d', A.name, ch), ...
                      fullfile(od, ['acg_ccg_' A.name '.png']), maxlag, binw);
            if ~isempty(spk)
                waveform_sim_matrix(cc, spk, sprintf('%s ch%d  mean-waveform correlation', A.name, ch), ...
                                    fullfile(od, ['waveform_xcorr_' A.name '.png']));
            end
            if ~isempty(gt.cc)
                spike_overlap_matrix(gt.cc, cc, sprintf('ch%d  ground truth vs %s  (spike-time Jaccard)', ch, A.name), ...
                                     fullfile(od, ['spike_overlap_' A.name '.png']));
            end
        end
    end
end

function corr_grid(cc, ttl, out_png, maxlag, binw)
    cls = sort(unique(cc(cc(:,1) > 0, 1)))';
    n = numel(cls);
    if n == 0, return; end
    lags = (-maxlag + binw/2):binw:(maxlag - binw/2);
    f = figure('Visible', 'off', 'Position', [80 80 max(700, 220*n) max(600, 200*n)], 'Color', 'w');
    tl = tiledlayout(f, n, n, 'Padding', 'compact', 'TileSpacing', 'compact');
    title(tl, [ttl '   (diag = ACG, off-diag = CCG; ms)'], 'Interpreter', 'none');
    for i = 1:n
        ti = sort(cc(cc(:,1) == cls(i), 2));
        for j = 1:n
            ax = nexttile(tl);
            tj = sort(cc(cc(:,1) == cls(j), 2));
            h = xcorr_hist(ti, tj, maxlag, binw, i == j);
            bar(ax, lags, h, 1, 'FaceColor', [0.2 0.3 0.6], 'EdgeColor', 'none');
            xlim(ax, [-maxlag maxlag]); set(ax, 'XTick', [-maxlag 0 maxlag]);
            if i == j
                title(ax, sprintf('c%d  ACG (n=%d)', cls(i), numel(ti)), 'FontSize', 8);
            else
                title(ax, sprintf('c%d x c%d', cls(i), cls(j)), 'FontSize', 8);
            end
            if j == 1, ylabel(ax, 'count', 'FontSize', 7); end
        end
    end
    save_fig(f, out_png);
end

% ----------------------------------------------------------------------
function temperature_diagram(tim_dir, nm, clu_par, out_png)
% Stand-alone SPC temperature diagram from data_<nm>.dg_01 (the SPC tree) -
% NOT the full wave_clus 3x5 figure. Two panels:
%   (1) cluster size vs temperature (log y), min_clus line, selected temp.
%   (2) how many clusters SURVIVE find_temp's selection at each temperature.
%       find_temp does NOT just count clusters >= min_clus - it only accepts a
%       cluster whose size JUMPS by > min_clus in one temperature step (a real
%       split, not gradual noise accretion), stops at the "elbow" (where the
%       main cluster shatters), and drops lower-T clusters that overlap a
%       higher-T one.  So a raw ">= min_clus" count over-reports what the
%       pipeline keeps; panel 2 plots the real find_temp result and marks the
%       actual final class count from the times file.
    dg = fullfile(tim_dir, ['data_' nm '.dg_01']);
    if ~exist(dg, 'file')
        d = dir(fullfile(tim_dir, 'data_*.dg_01'));
        if isempty(d), return; end
        dg = fullfile(tim_dir, d(1).name);
    end
    tree = load(dg);                                   % ascii matrix
    if isempty(tree) || size(tree, 2) < 5, return; end

    mintemp = clu_par.mintemp; maxtemp = clu_par.maxtemp;
    tstep = clu_par.tempstep;  minclus = clu_par.min_clus;
    c_ov = 0.7; elbow_min = 0.4;
    temp_sel = NaN; n_final = NaN;
    tf = fullfile(tim_dir, ['times_' nm '.mat']);
    if exist(tf, 'file')
        try
            G = load(tf, 'gui_status', 'par', 'cluster_class');
            if isfield(G, 'par')
                for fld = {'mintemp','maxtemp','tempstep','min_clus','c_ov','elbow_min'}
                    if isfield(G.par, fld{1}) && ~isempty(G.par.(fld{1}))
                        switch fld{1}
                            case 'mintemp',  mintemp = G.par.mintemp;
                            case 'maxtemp',  maxtemp = G.par.maxtemp;
                            case 'tempstep', tstep   = G.par.tempstep;
                            case 'min_clus', minclus = G.par.min_clus;
                            case 'c_ov',     c_ov    = G.par.c_ov;
                            case 'elbow_min',elbow_min = G.par.elbow_min;
                        end
                    end
                end
            end
            if isfield(G, 'gui_status') && isfield(G.gui_status, 'current_temp')
                temp_sel = mintemp + double(G.gui_status.current_temp) * tstep;
            end
            if isfield(G, 'cluster_class') && ~isempty(G.cluster_class)
                n_final = numel(setdiff(unique(G.cluster_class(:,1)), 0));
            end
        catch
        end
    end
    num_temp = min(size(tree, 1), floor((maxtemp - mintemp) / tstep));
    if num_temp < 2, return; end
    temps = mintemp + (1:num_temp)' * tstep;
    sizes = tree(1:num_temp, 5:end);                   % [temp x rank]
    nclus_raw = sum(sizes >= minclus, 2);

    % ---- replicate find_temp's cluster-selection matrix -------------------
    par_ft = struct('min_clus', minclus, 'c_ov', c_ov, 'elbow_min', elbow_min, ...
                    'mintemp', mintemp, 'tempstep', tstep);
    clu = [];
    lab = [dg '.lab'];
    if exist(lab, 'file'), try, clu = load(lab); catch, clu = []; end, end
    [nkeep, elbow_row, sel_tc] = find_temp_curve(tree, clu, par_ft);
    nkeep = nkeep(1:min(numel(nkeep), num_temp));
    if numel(nkeep) < num_temp && ~isempty(nkeep)
        nkeep(end+1:num_temp) = nkeep(find(~isnan(nkeep), 1, 'last'));
    end

    f = figure('Visible', 'off', 'Position', [100 100 1000 720], 'Color', 'w');
    tl = tiledlayout(f, 2, 1, 'Padding', 'loose', 'TileSpacing', 'loose');
    title(tl, sprintf('%s   SPC temperature diagram   (min\\_clus = %g)', nm, minclus), ...
          'Interpreter', 'none');

    ax = nexttile(tl); hold(ax, 'on');
    if ~isnan(elbow_row) && elbow_row <= num_temp
        xpatch = [temps(elbow_row) maxtemp maxtemp temps(elbow_row)];
        patch(ax, xpatch, [1e-2 1e-2 1e6 1e6], [0.92 0.92 0.92], 'EdgeColor', 'none');
    end
    plot(ax, temps, max(sizes, 0.5), '-', 'Color', [0.35 0.45 0.75]);
    yline(ax, minclus, 'k:', 'min\_clus');
    if ~isempty(sel_tc)
        for k = 1:size(sel_tc, 1)
            r = sel_tc(k, 1); cc = sel_tc(k, 2);
            if r <= num_temp && cc <= size(sizes, 2)
                plot(ax, temps(r), max(sizes(r, cc), 0.5), 'o', 'MarkerSize', 9, ...
                     'MarkerEdgeColor', [0.85 0.1 0.1], 'LineWidth', 1.6);
            end
        end
    end
    if isfinite(temp_sel), xline(ax, temp_sel, 'r-', 'selected T', 'LabelVerticalAlignment', 'bottom'); end
    set(ax, 'YScale', 'log'); grid(ax, 'on');
    xlim(ax, [mintemp maxtemp]); xlabel(ax, 'temperature'); ylabel(ax, 'cluster size (spikes)');
    title(ax, 'cluster size vs temperature (each line = Nth-largest cluster; \circ = kept by find\_temp)');

    ax = nexttile(tl); hold(ax, 'on');
    if ~isnan(elbow_row) && elbow_row <= num_temp
        yl = [0 max([3; nclus_raw(:) + 1])];
        patch(ax, [temps(elbow_row) maxtemp maxtemp temps(elbow_row)], ...
              [yl(1) yl(1) yl(2) yl(2)], [0.92 0.92 0.92], 'EdgeColor', 'none');
        text(ax, temps(elbow_row), yl(2)*0.9, ' elbow: main cluster shatters', ...
             'FontSize', 8, 'Color', [0.4 0.4 0.4]);
    end
    stairs(ax, temps, nclus_raw, '-', 'Color', [0.7 0.75 0.8], 'LineWidth', 1.2);
    stairs(ax, temps, nkeep, '-', 'Color', [0.2 0.5 0.3], 'LineWidth', 2);
    if isfinite(temp_sel), xline(ax, temp_sel, 'r-'); end
    if ~isnan(n_final)
        yline(ax, n_final, 'r--', sprintf('final: %d classes', n_final), ...
              'LabelHorizontalAlignment', 'left');
    end
    grid(ax, 'on'); xlim(ax, [mintemp maxtemp]);
    ylim(ax, [0 max([3; nclus_raw(:) + 1])]);
    xlabel(ax, 'temperature'); ylabel(ax, '# clusters');
    legend(ax, {'raw: size \geq min\_clus at that T', 'find\_temp: committed clusters (cumulative)'}, ...
           'Location', 'northwest', 'FontSize', 8);
    title(ax, 'clusters SPC proposes (grey) vs clusters find\_temp keeps (green) - green = the result');

    save_fig(f, out_png);
end

function [nkeep, elbow_row, sel_tc] = find_temp_curve(tree, clu, par)
% Port of clustering/find_temp.m that also returns nkeep = the CUMULATIVE count
% of clusters find_temp has committed to by each temperature (so it rises to the
% final class count and stays flat), the elbow row, and the finally-selected
% [temp_row, cluster_rank] pairs (sel_tc). Mirrors that file line-for-line so
% the panel matches the pipeline; the overlap-removal pass is skipped when the
% .dg_01.lab labels (clu) are unavailable.
    min_clus = par.min_clus; c_ov = par.c_ov; elbow_min = par.elbow_min;
    y = tree(1:end, 5);
    maxdiff = max(diff(tree(1:end, 6:end)), [], 2); maxdiff(maxdiff < 0) = 0;
    prop = (y(2:end) + maxdiff .* (maxdiff > 0)) ./ y(1:end-1);
    aux = find(prop < elbow_min, 1, 'first') + 1;
    if ~isempty(aux) && par.mintemp == 0 && aux == 2
        aux = find(prop(2:end) < elbow_min, 1, 'first') + 2;
    end
    tr = tree(1:end-1, 5:end);
    clus = zeros(size(tr)); clus(tr >= min_clus) = 1;
    dt = diff(tr);
    clus = clus & [ones(1, size(clus, 2)); dt > min_clus];
    for ii = 1:size(clus, 1)
        d = find(clus(ii, :), 1, 'last');
        if ~isempty(d), clus(ii, 1:d) = 1; end
    end
    elbow_row = size(tr, 1);
    if ~isempty(aux), clus(aux:end, :) = 0; elbow_row = aux; end
    if ~isempty(clu) && size(clu, 1) >= size(tr, 1) + 1
        cl = clu(1:end-1, 3:end) + 1;
        for ti = size(clus, 1):-1:1
            detect = find(clus(ti, :));
            for ci = 1:numel(detect)
                a = (cl(ti, :) == detect(ci));
                for tj = ti-1:-1:1
                    rem = find(clus(tj, :));
                    for j = 1:numel(rem)
                        b = (cl(tj, :) == rem(j));
                        if nnz(a & b) / max(min(nnz(b), nnz(a)), 1) >= c_ov
                            clus(tj, rem(j)) = 0;
                        end
                    end
                end
            end
        end
    end
    % cumulative: "# clusters find_temp has committed to by temperature T"
    nkeep = [cumsum(sum(clus, 2)); NaN];
    nkeep = nkeep(1:size(tree, 1));
    [tt, cc] = find(clus);
    sel_tc = sortrows([tt, cc]);
end

% ----------------------------------------------------------------------
function ST = feature_space_plot(arm_info, channels, NSx, hs_dir, gt_cc_cache, cmp_dir)
% One panel per channel, KS statistic per wavelet coefficient ranked descending.
% The comparison is GROUND-TRUTH feature space vs ALL-DETECTED feature space.
% Curves (spike count in the legend):
%   handselect (ground-truth units)   the hand sort's ACCEPTED units, i.e. the
%                          grapes classN reconstruction (`gt_cc_cache(ch).cc`,
%                          stimulus-windowed + clustered). Spike TIMES are
%                          matched to `times_<nm>.mat` in hs_dir to pull the
%                          waveforms; the times file's own cluster_class is
%                          NOT used - only its `spikes` + spike times.
%   handselect (all detected)         wave_features over EVERY waveform in that
%                          hand-select times file - the full detection, no sort.
%   nomask_recluster       the arm's OWN Do_features KS from
%                          <arm>/<nm>_spikes.mat, verbatim - all detected spikes
%                          of the new-trial run that SPC clustered.
%   masked_recluster       same, after within_channel + refract_viol
%                          (the kept subset). nomask -> masked = rejection effect.
% Every hand-select curve uses a COMMON par (the nomask arm's) so the only
% variable between curves is the spike population, not the wavelet transform.
    ST = table();
    for ci = 1:numel(channels)
        ch = channels(ci); nm = output_name_for(NSx, ch);
        curves = {};   % {label, ks, colour, nspk, style}

        % ---- recluster arms: real Do_features KS, verbatim ----
        pal = struct('nomask_recluster', [0.85 0.1 0.1], 'masked_recluster', [0.1 0.4 0.85]);
        common_par = [];
        for ai = 1:numel(arm_info)
            A = arm_info(ai);
            if strcmp(A.name, 'handselect'), continue; end
            Sp = safe_load(fullfile(A.dir, [nm '_spikes.mat']));
            if strcmp(A.name, 'nomask_recluster') && isfield(Sp, 'par'), common_par = Sp.par; end
            % rescue arm reuses the masked features block verbatim (rescue_spikes
            % does not recompute wave_features) - skip the duplicate line.
            if strcmp(A.name, 'masked_recluster_rescue'), continue; end
            if ~isfield(Sp, 'features') || ~isfield(Sp.features, 'ks'), continue; end
            n = NaN;
            if isfield(Sp, 'index'), n = numel(Sp.index);
            elseif isfield(Sp, 'spikes'), n = size(Sp.spikes, 1); end
            c = [0.3 0.3 0.3]; if isfield(pal, A.name), c = pal.(A.name); end
            curves{end+1} = {A.name, Sp.features.ks, c, n, '-'};
        end

        % ---- handselect: waveforms + spike times ONLY from the times file ----
        HT = safe_load(fullfile(hs_dir, ['times_' nm '.mat']));
        par = common_par;
        if isempty(par) && isfield(HT, 'par'), par = HT.par; end
        if isfield(HT, 'spikes') && ~isempty(par) && size(HT.spikes, 1) >= 20
            % ground-truth units: grapes classN reconstruction, matched by time
            gc = []; if isKey(gt_cc_cache, ch), gc = gt_cc_cache(ch); end
            if ~isempty(gc) && ~isempty(gc.cc) && isfield(HT, 'cluster_class')
                mi = nearest_match(gc.cc(:,2), HT.cluster_class(:,2), 0.5); mi = mi(mi > 0);
                if numel(mi) >= 20
                    kg = kswf(HT.spikes(mi, :), par);
                    if ~isempty(kg)
                        curves{end+1} = {'handselect (ground-truth units)', kg, [0 0 0], numel(mi), '--'};
                    end
                end
            end
            % all detected: every hand-select waveform, no sort applied
            ka = kswf(HT.spikes, par);
            if ~isempty(ka)
                curves{end+1} = {'handselect (all detected)', ka, [0.5 0.5 0.5], size(HT.spikes,1), ':'};
            end
        end

        if numel(curves) < 2, continue; end
        f = figure('Visible', 'off', 'Position', [100 100 1000 580], 'Color', 'w');
        ax = axes(f); hold(ax, 'on');
        hh = []; leg = {};
        for q = 1:numel(curves)
            ksd = sort(reshape(curves{q}{2}, 1, []), 'descend');
            hh(end+1) = plot(ax, 1:numel(ksd), ksd, curves{q}{5}, 'Color', curves{q}{3}, ...
                             'LineWidth', 1.6);
            leg{end+1} = sprintf('%s  (n=%d spk, maxKS %.3f)', curves{q}{1}, ...
                                 round(curves{q}{4}), max(ksd));
            ST = [ST; table(string(curves{q}{1}), ch, round(curves{q}{4}), max(ksd), ...
                            mean(ksd(1:min(10, numel(ksd)))), ...
                'VariableNames', {'set','channel_id','n_spikes','ks_max','ks_top10_mean'})];
        end
        hold(ax, 'off'); grid(ax, 'on');
        legend(ax, hh, leg, 'Interpreter', 'none', 'Location', 'northeast');
        xlabel(ax, 'wavelet coefficient (ranked)'); ylabel(ax, 'KS statistic');
        title(ax, {sprintf('ch%d  feature-space discriminability (KS per coefficient)', ch), ...
              'ground-truth units vs all-detected  |  solid = recluster arms  dashed/dotted = hand-select'});
        save_fig(f, fullfile(cmp_dir, sprintf('feature_space_ch%d.png', ch)));
    end
    if ~isempty(ST)
        ST = movevars(ST, {'set', 'channel_id'}, 'Before', 1);
        writetable(ST, fullfile(cmp_dir, 'feature_space.csv'));
    end
end

function ks = kswf(spikes, par)
    ks = [];
    try
        fq = wave_features(spikes, par);
        ks = fq.ks;
    catch
    end
end

function S = safe_load(f)
    S = struct();
    if exist(f, 'file')
        try
            S = load(f);
        catch
        end
    end
end

% ----------------------------------------------------------------------
function patch_rescue_times(tf)
% rescue_spikes grows cluster_class / spikes / inspk with the re-admitted
% spikes but leaves 'forced' and 'gui_status.original_classes' at the
% pre-rescue length, so the wave_clus GUI crashes on load
% (clustering_results(:,4) = saved_gui_status : size mismatch). Pad them.
% 'ipermut' must NOT be padded: ipermut(j) is the spike row of column j+2 of
% the SPC .lab file, so it keeps its pre-rescue length and only its row
% indices are remapped (a padded ipermut makes the GUI index past clu).
    if ~exist(tf, 'file'), return; end
    S = load(tf);
    if ~isfield(S, 'cluster_class'), return; end
    n = size(S.cluster_class, 1);
    changed = false;
    if isfield(S, 'forced') && numel(S.forced) < n
        S.forced = [S.forced(:); true(n - numel(S.forced), 1)];   % rescued = force-assigned
        changed = true;
    end
    if isfield(S, 'ipermut_pre_rescue') && isfield(S, 'cluster_class_pre_rescue') && ...
            ~isempty(S.ipermut_pre_rescue)
        % rebuild from the backup: pre-rescue row -> current row, by spike time
        [found, new_row] = ismember(S.cluster_class_pre_rescue(:, 2), S.cluster_class(:, 2));
        if all(found(S.ipermut_pre_rescue(:)))
            ip = reshape(new_row(S.ipermut_pre_rescue(:)), size(S.ipermut_pre_rescue));
            if ~isfield(S, 'ipermut') || ~isequal(S.ipermut, ip)
                S.ipermut = ip;
                changed = true;
            end
        end
    end
    if isfield(S, 'gui_status') && isstruct(S.gui_status) && ...
            isfield(S.gui_status, 'original_classes') && numel(S.gui_status.original_classes) < n
        pad = n - numel(S.gui_status.original_classes);
        S.gui_status.original_classes = [S.gui_status.original_classes(:); ...
                                         S.cluster_class(end - pad + 1:end, 1)];
        changed = true;
    end
    if changed
        save(tf, '-struct', 'S');
        fprintf('[recval]   patched %s for the wave_clus GUI (forced/gui_status/ipermut -> %d rows).\n', tf, n);
    end
end

function h = xcorr_hist(t1, t2, maxlag, binw, is_acg)
% Histogram of (t2 - t1) for pairs within +/- maxlag ms. For an ACG the
% self-pairs (lag 0) are excluded.
    edges = -maxlag:binw:maxlag;
    h = zeros(1, numel(edges) - 1);
    if isempty(t1) || isempty(t2), return; end
    n2 = numel(t2); j0 = 1;
    for i = 1:numel(t1)
        while j0 <= n2 && t2(j0) < t1(i) - maxlag, j0 = j0 + 1; end
        k = j0;
        while k <= n2 && t2(k) <= t1(i) + maxlag
            d = t2(k) - t1(i);
            if ~(is_acg && d == 0)
                b = floor((d + maxlag) / binw) + 1;
                if b >= 1 && b <= numel(h), h(b) = h(b) + 1; end
            end
            k = k + 1;
        end
    end
end

function waveform_sim_matrix(cc, spk, ttl, out_png)
    cls = sort(unique(cc(cc(:,1) > 0, 1)))';
    n = numel(cls);
    if n < 1 || size(spk, 1) ~= size(cc, 1), return; end
    mw = zeros(n, size(spk, 2));
    for i = 1:n, mw(i, :) = mean(spk(cc(:,1) == cls(i), :), 1); end
    R = corrcoef(mw');
    if n == 1, R = 1; end
    f = figure('Visible', 'off', 'Position', [100 100 max(420, 90*n + 220) max(380, 90*n + 160)], 'Color', 'w');
    ax = axes(f);
    imagesc(ax, R, [-1 1]); colorbar(ax); colormap(ax, diverging_cmap()); axis(ax, 'image');
    xticks(ax, 1:n); yticks(ax, 1:n);
    xticklabels(ax, compose('c%d', cls)); yticklabels(ax, compose('c%d', cls));
    for i = 1:n
        for j = 1:n
            text(ax, j, i, sprintf('%.2f', R(i, j)), 'HorizontalAlignment', 'center', 'FontSize', 8);
        end
    end
    title(ax, ttl, 'Interpreter', 'none');
    save_fig(f, out_png);
end

function spike_overlap_matrix(gt_cc, ar_cc, ttl, out_png)
    gcl = sort(unique(gt_cc(gt_cc(:,1) > 0, 1)))';
    acl = sort(unique(ar_cc(ar_cc(:,1) > 0, 1)))';
    ng = numel(gcl); na = numel(acl);
    if ng == 0 || na == 0, return; end
    tol = 0.5;
    [gt, o1] = sort(gt_cc(:,2)); gk = gt_cc(o1,1);
    [at, o2] = sort(ar_cc(:,2)); ak = ar_cc(o2,1);
    N = zeros(ng, na); used = false(numel(at), 1); j = 1;
    for i = 1:numel(gt)
        while j <= numel(at) && at(j) < gt(i) - tol, j = j + 1; end
        k = j; best = -1; bd = tol + 1;
        while k <= numel(at) && at(k) <= gt(i) + tol
            if ~used(k) && abs(at(k) - gt(i)) < bd, bd = abs(at(k) - gt(i)); best = k; end
            k = k + 1;
        end
        if best > 0
            used(best) = true;
            gi = find(gcl == gk(i), 1); ai2 = find(acl == ak(best), 1);
            N(gi, ai2) = N(gi, ai2) + 1;
        end
    end
    ng_v = arrayfun(@(k) nnz(gk == k), gcl);
    na_v = arrayfun(@(k) nnz(ak == k), acl);
    J = N ./ (ng_v(:) + na_v(:)' - N);
    f = figure('Visible', 'off', 'Position', [100 100 max(460, 90*na + 240) max(380, 80*ng + 180)], 'Color', 'w');
    ax = axes(f);
    imagesc(ax, J, [0 1]); colorbar(ax); colormap(ax, parula); axis(ax, 'image');
    xticks(ax, 1:na); yticks(ax, 1:ng);
    xticklabels(ax, compose('arm c%d', acl)); yticklabels(ax, compose('GT c%d', gcl));
    set(ax, 'TickLabelInterpreter', 'none');
    for i = 1:ng
        for j2 = 1:na
            text(ax, j2, i, sprintf('%.2f', J(i, j2)), 'HorizontalAlignment', 'center', 'FontSize', 8);
        end
    end
    title(ax, ttl, 'Interpreter', 'none');
    xlabel(ax, 'arm cluster'); ylabel(ax, 'ground-truth class');
    save_fig(f, out_png);
end

% ----------------------------------------------------------------------
function report_path = write_report(out, arm_info, o, cmp_dir)
% Big-scale HTML page: one section per (channel, ground-truth class) with the
% match / composition / response / quality tables, the key figures, and a
% stimulus grid (thumbnail + zscore per arm, colour-coded) so the picture-level
% impact of the sorting differences is visible at a glance.
    report_path = fullfile(o.out_dir, 'report.html');
    pics_dir = resolve_pics_dir(arm_info);
    thumb_dir = fullfile(cmp_dir, '_thumb');
    if ~exist(thumb_dir, 'dir'), mkdir(thumb_dir); end

    fid = fopen(report_path, 'w');
    c = onCleanup(@() fclose(fid));
    fprintf(fid, ['<!doctype html><meta charset="utf-8"><title>recluster validation</title>' ...
        '<style>body{font:13px/1.4 -apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#111}' ...
        'h2{border-bottom:2px solid #333;margin-top:40px}h3{margin:18px 0 6px}' ...
        'table{border-collapse:collapse;margin:8px 0}td,th{border:1px solid #bbb;padding:3px 7px;text-align:right}' ...
        'th{background:#eee}td.l,th.l{text-align:left}img{max-width:100%%;border:1px solid #ddd;margin:4px 0}' ...
        '.grid td{padding:1px 4px}.grid img{height:46px;width:46px;object-fit:cover;border:0;margin:0}' ...
        '.small{color:#666;font-size:11px}</style>']);
    fprintf(fid, '<h1>Re-cluster vs hand-select</h1><p class=small>%s<br>arms: %s<br>stim selection: %s</p>', ...
        strrep(o.out_dir,'&','&amp;'), strjoin({arm_info.name}, ', '), stim_sel_str(o));

    keys = fieldnames(out.compare)';
    fprintf(fid, '<p>Sections: %s</p>', strjoin(cellfun(@(k) sprintf('<a href="#%s">%s</a>',k,k), keys, ...
        'UniformOutput', false), ' &middot; '));

    seen_ch = [];
    for kk = keys
        cmp = out.compare.(kk{1});
        ch = cmp.channel; gc = cmp.gt_class;
        fprintf(fid, '<h2 id="%s">ch%d &nbsp; ground-truth class %d</h2>', kk{1}, ch, gc);

        % ---- once per channel: cluster make-up (temperature maps + correlograms) ----
        if ~ismember(ch, seen_ch)
            seen_ch(end+1) = ch;
            fprintf(fid, '<h3>cluster make-up (channel %d)</h3>', ch);
            fp = sprintf('comparisons/feature_space_ch%d.png', ch);
            if exist(fullfile(o.out_dir, fp), 'file'), fprintf(fid, '<img src="%s">', fp); end
            for ii = 1:numel(arm_info)
                a = arm_info(ii).name;
                for img = {sprintf('comparisons/temperature_ch%d_%s.png', ch, a), ...
                           sprintf('comparisons/cluster_comparison_ch%d/acg_ccg_%s.png', ch, a), ...
                           sprintf('comparisons/cluster_comparison_ch%d/waveform_xcorr_%s.png', ch, a), ...
                           sprintf('comparisons/cluster_comparison_ch%d/spike_overlap_%s.png', ch, a)}
                    if exist(fullfile(o.out_dir, img{1}), 'file')
                        fprintf(fid, '<img src="%s">', img{1});
                    end
                end
            end
        end

        % ---- match headline + where GT spikes go + arm composition ----
        html_table(fid, filt(out, 'match_all', ch, gc), 'match (per arm)');
        html_table(fid, filt(out, 'match_detail', ch, gc), 'where this class''s spikes go');
        html_table(fid, filt(out, 'match_composition', ch, gc), 'what the matched arm cluster is made of');
        html_table(fid, filt(out, 'summary', ch, gc), 'response outcome (per arm)');
        if isfield(out,'sorting_metrics')
            html_table(fid, filt(out, 'sorting_metrics', ch, gc), 'cluster quality (matched class, per arm)');
        end

        % ---- figures ----
        pd = sprintf('comparisons/%s_plots', kk{1});
        imgs = {sprintf('comparisons/dashboard_%s.png', kk{1}), ...
                [pd '/waveform_overlay.png'], [pd '/heatmap_zscore.png'], ...
                [pd '/heatmap_delta_zscore.png'], [pd '/change_zscore.png']};
        for ii = 1:numel(arm_info)
            if strcmp(arm_info(ii).name, 'handselect'), continue; end
            imgs{end+1} = sprintf('comparisons/class_match_ch%d_%s.png', ch, ...
                                  matlab.lang.makeValidName(arm_info(ii).name));
        end
        for img = imgs
            if exist(fullfile(o.out_dir, img{1}), 'file')
                fprintf(fid, '<img src="%s">', img{1});
            end
        end

        % ---- stimulus grid: thumbnail + zscore per arm ----
        stim_grid(fid, cmp, pics_dir, thumb_dir);
    end
    fprintf(fid, '<hr><p class=small>generated %s</p>', char(datetime('now')));
end

function stim_grid(fid, cmp, pics_dir, thumb_dir)
    Z = cmp.wide.zscore;
    ord = (1:numel(cmp.stim_number))';   % caller's stim_list order, verbatim
    labels = cmp.arm_labels;
    lo = min(Z(:), [], 'omitnan'); hi = max(Z(:), [], 'omitnan');
    fprintf(fid, ['<h3>stimulus grid <span class=small>(stim_list order; ' ...
        'cell = zscore, bg = value, &bull; = responsive)</span></h3>' ...
        '<table class=grid><tr><th class=l>stim</th><th></th>']);
    for k = 1:numel(labels), fprintf(fid, '<th>%s</th>', esc(labels{k})); end
    fprintf(fid, '</tr>');
    for ii = ord(:)'
        th = stim_thumb(cmp.stim_name(ii), pics_dir, thumb_dir);
        rel = '';
        if ~isempty(th), rel = ['comparisons/_thumb/' th]; end
        fprintf(fid, '<tr><td class=l>%d %s</td><td>%s</td>', cmp.stim_number(ii), ...
            esc(char(cmp.stim_name(ii))), ...
            iif(isempty(rel), '', sprintf('<img src="%s">', rel)));
        for k = 1:numel(labels)
            z = cmp.wide.zscore(ii, k);
            dot = ''; if k <= size(cmp.responsive,2) && cmp.responsive(ii,k), dot = ' &bull;'; end
            fprintf(fid, '<td style="background:%s">%s%s</td>', heat_rgb(z, lo, hi), num2str_(z), dot);
        end
        fprintf(fid, '</tr>');
    end
    fprintf(fid, '</table>');
end

function T = filt(out, which, ch, gc)
    T = table();
    switch which
        case 'match_all'
            for fn = fieldnames(out.match)'
                T = [T; out.match.(fn{1})(out.match.(fn{1}).channel_id==ch & ...
                                          out.match.(fn{1}).gt_class==gc, :)];
            end
        case 'match_detail'
            if isfield(out,'match_detail') && ~isempty(out.match_detail)
                T = out.match_detail(out.match_detail.channel_id==ch & out.match_detail.gt_class==gc, :);
            end
        case 'match_composition'
            if isfield(out,'match_composition') && ~isempty(out.match_composition)
                T = out.match_composition(out.match_composition.channel_id==ch & ...
                                          out.match_composition.gt_class==gc, :);
            end
        case 'summary'
            T = out.summary(out.summary.channel_id==ch & out.summary.gt_class==gc, :);
        case 'sorting_metrics'
            if isfield(out,'sorting_metrics') && ~isempty(out.sorting_metrics)
                T = out.sorting_metrics(out.sorting_metrics.channel_id==ch & ...
                                        out.sorting_metrics.gt_class==gc, :);
            end
    end
end

function html_table(fid, T, caption)
    if isempty(T) || height(T)==0, return; end
    vn = T.Properties.VariableNames;
    fprintf(fid, '<h3>%s</h3><table><tr>', esc(caption));
    for v = vn, fprintf(fid, '<th class=l>%s</th>', esc(v{1})); end
    fprintf(fid, '</tr>');
    for r = 1:height(T)
        fprintf(fid, '<tr>');
        for v = vn
            x = T.(v{1})(r);
            if isnumeric(x)
                s = num2str_(x);
            else
                s = esc(char(string(x)));
            end
            fprintf(fid, '<td>%s</td>', s);
        end
        fprintf(fid, '</tr>');
    end
    fprintf(fid, '</table>');
end

function name = stim_thumb(stim_name, pics_dir, thumb_dir)
    name = '';
    sn = char(stim_name);
    if isempty(sn) || isempty(pics_dir), return; end
    src = '';
    cand = fullfile(pics_dir, sn);
    if exist(cand, 'file'), src = cand;
    else
        [~, b] = fileparts(sn);
        d = dir(fullfile(pics_dir, [b '.*']));
        if ~isempty(d), src = fullfile(d(1).folder, d(1).name); end
    end
    if isempty(src), return; end
    [~, b, e] = fileparts(src);
    name = matlab.lang.makeValidName(b);
    name = [name e];
    dst = fullfile(thumb_dir, name);
    if ~exist(dst, 'file')
        ok = false;
        try
            im = imread(src);
            im = imresize(im, [64 64]);
            imwrite(im, dst);
            ok = true;
        catch
        end
        if ~ok
            try
                copyfile(src, dst);
            catch
                name = '';
            end
        end
    end
end

function s = heat_rgb(z, lo, hi)
    if ~isfinite(z), s = '#f0f0f0'; return; end
    if z >= 0
        t = min(1, z / max(hi, eps));
        c = round(255 * (1 - 0.75*t));
        s = sprintf('#ff%02x%02x', c, c);
    else
        t = min(1, z / min(lo, -eps));
        c = round(255 * (1 - 0.75*t));
        s = sprintf('#%02x%02xff', c, c);
    end
end

function d = resolve_pics_dir(arm_info)
    d = '';
    for i = 1:numel(arm_info)
        try
            S = load(arm_info(i).metrics, 'meta');
            if isfield(S, 'meta') && isfield(S.meta, 'pics_dir') && isfolder(S.meta.pics_dir)
                d = S.meta.pics_dir; return;
            end
        catch
        end
    end
end

function s = stim_sel_str(o)
    if ~isempty(o.scr), s = sprintf('screening block %d', o.scr);
    elseif isnumeric(o.stim_list), s = sprintf('%d explicit stimuli', numel(o.stim_list));
    else, s = 'all screening blocks (union)';
    end
end

function s = num2str_(x)
    if ~isnumeric(x) || ~isscalar(x), s = esc(char(string(x))); return; end
    if ~isfinite(x), s = ''; return; end
    if x == round(x), s = sprintf('%d', x);
    elseif abs(x) >= 100, s = sprintf('%.0f', x);
    else, s = sprintf('%.2f', x);
    end
end

function s = esc(x)
    s = char(string(x));
    s = strrep(s, '&', '&amp;'); s = strrep(s, '<', '&lt;'); s = strrep(s, '>', '&gt;');
end

function r = iif(cond, a, b)
    if cond, r = a; else, r = b; end
end

function make_heatmap(M, row_lab, col_lab, ttl, diverging, out_png)
    nr = size(M, 1);
    rlw = max(cellfun(@numel, cellstr(row_lab)));
    fs = 8; if nr > 45, fs = 6; end
    fig_w = max(900, 170*numel(col_lab) + 8*rlw);
    fig_h = min(1500, max(320, 22*nr + 170));
    f = figure('Visible', 'off', 'Position', [100 100 fig_w fig_h], 'Color', 'w');
    ax = axes(f);
    imagesc(ax, M, 'AlphaData', ~isnan(M)); colorbar(ax);
    v = M(isfinite(M));
    if diverging
        mx = quantile(abs(v), 0.98); if isempty(mx) || mx <= 0, mx = max(abs(v)); end
        if ~isempty(mx) && mx > 0, set_clim(ax, [-mx mx]); end
        colormap(ax, diverging_cmap());
    else
        rl = robust_lims(v); if ~isempty(rl), set_clim(ax, rl); end
        colormap(ax, parula);
    end
    xticks(ax, 1:numel(col_lab)); xticklabels(ax, col_lab);
    yticks(ax, 1:nr); yticklabels(ax, row_lab);
    set(ax, 'TickLabelInterpreter', 'none', 'FontSize', fs);
    title(ax, ttl, 'Interpreter', 'none');
    L = min(0.5, 0.04 + (7*rlw)/fig_w);
    ax.Position = [L, 0.16, max(0.3, 0.82 - L), 0.72];
    save_fig(f, out_png);
end

function set_clim(ax, lims)
    try
        clim(ax, lims);
    catch
        caxis(ax, lims);
    end
end

function lims = robust_lims(v)
    lims = []; v = v(isfinite(v));
    if numel(v) < 3, return; end
    q = quantile(v, [0.02 0.98]);
    if q(2) <= q(1), q = [min(v) max(v)]; if q(2) <= q(1), return; end, end
    pad = 0.08 * (q(2) - q(1)); lims = [q(1)-pad, q(2)+pad];
end

function save_fig(f, out_png)
    set(f, 'Color', 'w');
    try
        print(f, out_png, '-dpng', '-r130');
    catch
        try
            exportgraphics(f, out_png, 'Resolution', 130, 'BackgroundColor', 'white');
        catch
        end
    end
    close(f);
end

function m = diverging_cmap()
    t = linspace(0, 1, 128)';
    m = [[t, t, ones(128,1)]; [ones(127,1), flipud(t(2:end)), flipud(t(2:end))]];
end

function s = shorten(c)
    s = cellfun(@(x) regexprep(x, '\s+', '_'), cellstr(c), 'UniformOutput', false);
    s = matlab.lang.makeUniqueStrings(s);
end

% ======================================================================
% small shared helpers
% ======================================================================
function stem = grapes_stem_for(use_blanks, circ)
    if use_blanks && circ,  stem = 'grapes_blanks_circ';
    elseif use_blanks,      stem = 'grapes_blanks';
    else,                   stem = 'grapes';
    end
end

function nm = output_name_for(NSx, ch)
    pos = find(arrayfun(@(x) x.chan_ID == ch, NSx), 1);
    if isempty(pos), nm = ''; else, nm = NSx(pos).output_name; end
end

function copy_channel_files(src_dir, dest_dir, channels, NSx, suffix)
    for ch = channels(:)'
        nm = output_name_for(NSx, ch);
        if isempty(nm), warning('chan %d not in NSx', ch); continue; end
        f = fullfile(src_dir, [nm suffix]);
        if exist(f, 'file'), copyfile(f, dest_dir);
        else, warning('validate_recluster_vs_handselect: %s not found', f);
        end
    end
end

function copy_if(f, dest)
    if exist(f, 'file'), copyfile(f, dest); end
end

function strip_stale_grapes(D, names)
    % Remove ONLY the specific grapes file(s) about to be rebuilt in this
    % arm-output folder, so do_structure_* / plot_grapes_as_online don't append
    % to or read a leftover from an earlier run. Never touches the user's
    % source hand-select folder (this only ever runs on RECLUSTER_VALIDATION_*
    % subfolders), and never a blanket 'grapes*' wipe.
    if nargin < 2 || isempty(names)
        names = {'grapes.mat', 'grapes_blanks.mat', 'grapes_blanks_circ.mat', ...
                 'grapes_rescue.mat', 'grapes_blanks_rescue.mat', 'grapes_blanks_circ_rescue.mat'};
    end
    names = cellstr(names);
    for i = 1:numel(names)
        f = fullfile(D, names{i});
        if exist(f, 'file'), delete(f); end
    end
end

function snapshot(spk_dir, tim_dir, dest, channels, NSx)
    if ~exist(dest, 'dir'), mkdir(dest); end
    for ch = channels(:)'
        nm = output_name_for(NSx, ch);
        if isempty(nm), continue; end
        copy_if(fullfile(spk_dir, [nm '_spikes.mat']), dest);
        copy_if(fullfile(tim_dir, ['times_' nm '.mat']), dest);
        copy_if(fullfile(tim_dir, ['temperature_' nm '.png']), dest);
    end
    copy_if(fullfile(tim_dir, 'templates_wc_offline.mat'), dest);
    strip_stale_grapes(dest);   % grapes here must be rebuilt from THIS times copy
end

% ======================================================================
% grapes -> cluster_class reconstruction (grapes is the sole ground truth)
% ======================================================================
function omap = build_onset_map(root)
% snum -> [stimulus onset times, ms], accumulated in the exact order
% update_grapes / update_grapes_blanks append trials to raster.stim{snum}.
% Replicates update_grapes's onset lookup verbatim, quirk included:
%   2D pics_onset -> pics_onset(terna(:,1), terna(:,3)) then linear 1:length.
    omap = containers.Map('KeyType', 'double', 'ValueType', 'any');
    S = load(fullfile(root, 'stimulus.mat'));       stimulus = S.stimulus;
    F = load(fullfile(root, 'finalevents.mat'));    pics_onset = F.pics_onset;
    E = load(fullfile(root, 'experiment_properties_online3.mat'), 'scr_end_cell');
    Nscr = min(numel(stimulus), numel(E.scr_end_cell));
    for scri = 1:Nscr
        po = pics_onset{scri};
        st = stimulus{scri};
        for j = 1:numel(st)
            snum  = st(j).ID;
            terna = st(j).terna_onset;
            if ndims(po) == 3
                ts = po(sub2ind(size(po), terna(:,1), terna(:,2), terna(:,3)));
            else
                M  = po(terna(:,1), terna(:,3));
                ts = M(1:length(M));
            end
            ts = ts(:).';
            if ~isKey(omap, snum), omap(snum) = []; end
            omap(snum) = [omap(snum), ts];
        end
    end
end

function rc = reconstruct_cc_from_grapes(grapes_file, ch, omap)
% Rebuild a standard [class time_ms] cluster_class for one channel from a grapes
% raster, using only the POST-onset spikes (rel >= 0) so the result is
% identical whether the grapes is the plain or the blanks flavour (validated:
% plain vs blanks reconstructions match spike-for-spike). Spikes that fall in
% two overlapping stimulus windows are de-duplicated. No circshift handling is
% needed (circshift only touches the pre-onset blank spikes, which are dropped).
    rc = struct('cc', zeros(0,2), 'classes', [], 'ok', false, ...
                'n_by_class', [], 'n_trial_mismatch', 0, 'file', grapes_file);
    try
        G = load(grapes_file);
    catch
        return;
    end
    cf = ['chan' num2str(ch)];
    if ~isfield(G, 'rasters') || ~isfield(G.rasters, cf), return; end
    R = G.rasters.(cf);
    fn = fieldnames(R);
    cls = sort(cellfun(@(x) sscanf(x, 'class%d'), fn(~cellfun('isempty', ...
             regexp(fn, '^class\d+$', 'once')))))';
    rc.classes = cls;
    allt = []; allc = []; mism = 0;
    for ci = 1:numel(cls)
        C = sprintf('class%d', cls(ci));
        stimc = R.(C).stim;
        for snum = 1:numel(stimc)
            tr = stimc{snum};
            if isempty(tr), continue; end
            if ~iscell(tr), tr = num2cell(tr, 2); end
            if ~isKey(omap, snum), mism = mism + 1; continue; end
            O = omap(snum);
            if numel(O) ~= numel(tr), mism = mism + 1; continue; end
            for t = 1:numel(tr)
                rel = tr{t}(:).';
                rel = rel(rel >= 0 & rel < 9000);       % post-onset, drop 9999/10000
                if isempty(rel), continue; end
                allt = [allt, rel + O(t)];
                allc = [allc, repmat(cls(ci), 1, numel(rel))];
            end
        end
    end
    rc.n_trial_mismatch = mism;
    if isempty(allt)
        rc.ok = (mism == 0);
        return;
    end
    [allt, ord] = sort(allt); allc = allc(ord);
    % de-dup spikes shared by overlapping windows: same time (<=0.05 ms) & class
    keep = [true, ~(diff(allt) <= 0.05 & diff(allc) == 0)];
    allt = allt(keep); allc = allc(keep);
    rc.cc = [allc(:), allt(:)];   % standard cluster_class order: [class, time_ms]
    rc.n_by_class = arrayfun(@(k) nnz(allc == k), cls);
    rc.ok = (mism == 0) && ~isempty(cls);
end

function tf = times_matches_recon(times_file, rc, tol)
% True iff the cluster_class in times_file is the SAME clustering as the
% grapes reconstruction rc: identical class set AND, for every class, >=70% of
% the reconstructed spikes fall on a same-class spike in the times file.
    tf = false;
    if isempty(rc.classes) || isempty(rc.cc), return; end
    try
        S = load(times_file, 'cluster_class');
        cc = S.cluster_class;                       % [class time_ms]
    catch
        return;
    end
    tcls = sort(unique(cc(cc(:,1) > 0, 1)))';
    if ~isequal(tcls, rc.classes(:)'), return; end
    for k = rc.classes(:)'
        rt = sort(rc.cc(rc.cc(:,1) == k, 2));       % reconstructed class-k times
        if isempty(rt), continue; end
        tt = sort(cc(cc(:,1) == k, 2));             % times-file class-k times
        hit = 0; j = 1; nt = numel(tt);
        for i = 1:numel(rt)
            while j <= nt && tt(j) < rt(i) - tol, j = j + 1; end
            if j <= nt && abs(tt(j) - rt(i)) <= tol, hit = hit + 1; end
        end
        if hit / numel(rt) < 0.70, return; end
    end
    tf = true;
end

function run_in(d, fn)
    old = pwd;
    c = onCleanup(@() cd(old));
    cd(d);
    fn();
end
