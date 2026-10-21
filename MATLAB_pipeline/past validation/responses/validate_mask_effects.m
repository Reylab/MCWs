function out = validate_mask_effects(channels, varargin)
% VALIDATE_MASK_EFFECTS  Measure how within_channel + refract_viol (and the
% subsequent rescue) change stimulus responses, with the CLUSTERING HELD
% FIXED so class-vs-class comparisons are apples-to-apples across trials.
%
% The standard pipeline applies the QC masks BEFORE Do_features/Do_clustering,
% so masking changes the wavelet feature space and therefore the clustering -
% class 1 in a "masked" run is not necessarily class 1 in a "no-mask" run.
% This script clusters ONCE on the full spike set, then only removes / re-adds
% spikes from those fixed clusters:
%
%   1. copy a no-mask spikes folder            -> spikes_val_<ts>/
%   2. [opt] bundle_artifact
%   3. Do_features + Do_clustering(make_times, make_templates)   <- FULL set
%   4. snapshot -> <out>/nomask/ ; grapes mu+sorted ; compute_metrics_batch ; plot_grapes
%   5. within_channel + refract_viol           (spikes-folder masks)
%   6. apply_masks_to_times                    (drop masked spikes from the
%                                               existing cluster_class, no re-SPC)
%   7. snapshot -> <out>/masked/ ; grapes ; compute_metrics_batch ; plot_grapes
%   8. rescue_spikes                           (re-admit quarantined spikes vs
%                                               the unchanged cluster templates)
%   9. snapshot -> <out>/rescued/ ; do_structure_(mu|sorted|rescue) ;
%      compute_metrics_batch(rescue) ; plot_grapes(post) + plot_grapes(rescued-only)
%  10. collect_sorting_metrics -> comparisons/sorting_metrics.csv + per-channel PNG
%  11. characterize_grapes_responses in each trial folder
%  12. compare_grapes_response_trials across trials, per class + mu
%      (prefer_rescue mode: nomask & masked read the plain flavour, rescued
%       reads its <flavour>_rescue block)
%
% Required
%   channels   numeric chan_ID vector
%
% Name-value
%   'spikes_src'          ''     folder holding the no-mask *_spikes.mat
%                                 (default: newest spikes* under session root)
%   'use_blanks'          true
%   'circshiftblanks'     false
%   'seed'                0      Do_clustering seed
%   'run_features'        true
%   'run_bundle_artifact' false
%   'masks'  {'mask_non_quarantine','mask_non_refract'}  masks that define
%                                 "surviving" spikes for the masked trial
%   'stim_list'           'all'
%   'exp_type'            'RSVP_online'
%   'out_dir'             ''     default <root>/VALIDATION_<ts>
%   'do_plots'            true   run plot_grapes_as_online per trial
%   'do_compare'          true   run the characterize + response-compare stage
%   'do_sorting_metrics'  true   run compute_metrics_batch per trial and stack
%                                 the cluster-quality metrics across trials into
%                                 comparisons/sorting_metrics.{csv,mat} + PNGs
%   'compare_classes'     'all'  or a numeric vector of class numbers
%   'compare_trials' {'nomask','masked','rescued'}  which trials to build
%                                 grapes / plot / characterize / compare. The
%                                 masking + rescue STEPS always run when
%                                 'rescued' is requested (rescue needs them);
%                                 this just controls which trials get grapes
%                                 and enter the comparison. e.g.
%                                 {'nomask','rescued'} skips the masked trial.
%   'max_plot_stims'     40      forwarded to compare_grapes_response_trials
%                                 (above this: scatter plots instead of the
%                                  per-stim line plots + heatmaps)
%
% Output struct 'out': .out_dir .trials(struct name/dir/rescue/metrics)
%   .grapes_tag .cmp (compare_grapes_response_trials results per ch/unit)
%   .sorting_metrics (long table: trial,channel_id,cluster_id,<quality metrics>)

    % ---------- args ----------
    p = inputParser;
    addParameter(p, 'spikes_src', '', @ischar);
    addParameter(p, 'use_blanks', true, @islogical);
    addParameter(p, 'circshiftblanks', false, @islogical);
    addParameter(p, 'seed', 0, @isnumeric);
    addParameter(p, 'run_features', true, @islogical);
    addParameter(p, 'run_bundle_artifact', false, @islogical);
    addParameter(p, 'masks', {'mask_non_quarantine','mask_non_refract'}, @iscell);
    addParameter(p, 'stim_list', 'all');
    addParameter(p, 'exp_type', 'RSVP_online', @ischar);
    addParameter(p, 'out_dir', '', @ischar);
    addParameter(p, 'do_plots', true, @islogical);
    addParameter(p, 'do_compare', true, @islogical);
    addParameter(p, 'do_sorting_metrics', true, @islogical);
    addParameter(p, 'compare_classes', 'all');
    addParameter(p, 'compare_trials', {'nomask','masked','rescued'}, @iscell);
    addParameter(p, 'max_plot_stims', 40, @(x) isnumeric(x) && isscalar(x));
    parse(p, varargin{:});
    o = p.Results;

    valid_tr = {'nomask','masked','rescued'};
    o.compare_trials = lower(o.compare_trials(:)');
    bad = setdiff(o.compare_trials, valid_tr);
    assert(isempty(bad), 'validate_mask_effects: bad compare_trials: %s', strjoin(bad,', '));
    assert(~isempty(o.compare_trials), 'validate_mask_effects: compare_trials is empty.');
    want   = @(nm) ismember(nm, o.compare_trials);
    do_mask   = want('masked') || want('rescued');
    do_rescue = want('rescued');

    assert(isnumeric(channels) && ~isempty(channels), ...
           'validate_mask_effects: channels must be a numeric vector.');
    channels = double(channels(:)');

    repo = fileparts(fileparts(mfilename('fullpath')));   % .../MATLAB_pipeline
    addpath(genpath(repo));

    root = resolve_session_root();
    home = pwd;
    restore_cwd = onCleanup(@() cd(home));
    cd(root);

    ts = char(datetime('now','Format','yyyyMMdd_HHmm'));
    if isempty(o.out_dir), o.out_dir = fullfile(root, ['VALIDATION_' ts]); end
    if ~exist(o.out_dir, 'dir'), mkdir(o.out_dir); end
    grapes_stem = grapes_stem_for(o.use_blanks, o.circshiftblanks);
    gtag = matlab.lang.makeValidName(grapes_stem);
    fprintf('[validate] root=%s\n[validate] out=%s\n[validate] grapes=%s\n', ...
            root, o.out_dir, grapes_stem);

    load(fullfile(root, 'NSx.mat'), 'NSx');

    % ---------- 1. working spikes copy ----------
    src = o.spikes_src;
    if isempty(src)
        d = dir(fullfile(root, 'spikes*')); d = d([d.isdir]);
        assert(~isempty(d), 'no spikes* folder under %s; pass ''spikes_src''.', root);
        [~, i] = max([d.datenum]);
        src = fullfile(root, d(i).name);
    end
    spk_work = fullfile(root, ['spikes_val_' ts]);
    if ~exist(spk_work,'dir'), mkdir(spk_work); end
    copy_channel_files(src, spk_work, channels, NSx, '_spikes.mat');
    fprintf('[validate] working spikes: %s (from %s)\n', spk_work, src);

    % ---------- 2. bundle artifact ----------
    if o.run_bundle_artifact
        fprintf('[validate] bundle_artifact\n');
        bundle_artifact(channels);
    end

    % ---------- 3. features + clustering (FULL set) ----------
    tim_work = fullfile(root, ['times_val_' ts]);
    if ~exist(tim_work,'dir'), mkdir(tim_work); end
    if o.run_features
        fprintf('[validate] Do_features\n');
        Do_features(channels, 'parallel', true, 'folder', spk_work);
    end
    clu_par = struct('min_clus',20, 'max_spk',30000, 'mintemp',0, 'maxtemp',0.251, ...
                     'tempstep',0.01, 'max_std_templates',3, 'max_spikes_plot',5000);
    fprintf('[validate] Do_clustering (make_times, make_templates, seed=%g)\n', o.seed);
    Do_clustering(channels, 'parallel', true, 'make_times', true, ...
                  'make_templates', true, 'make_plots', false, 'par', clu_par, ...
                  'seed', o.seed, 'folder', spk_work, 'times_folder', tim_work);

    d_nomask = fullfile(o.out_dir, 'nomask');
    d_masked = fullfile(o.out_dir, 'masked');
    d_resc   = fullfile(o.out_dir, 'rescued');

    % ---------- 4. NOMASK trial ----------
    if want('nomask')
        snapshot(spk_work, tim_work, d_nomask, channels, NSx);
        trial_grapes(d_nomask, channels, o, root, false);
        trial_sorting_metrics(d_nomask, channels, o, root, false);
        if o.do_plots, trial_plot(d_nomask, channels, o, root, false, ''); end
    end

    % ---------- 5. masks (always run when masked or rescued wanted) ----------
    if do_mask
        fprintf('[validate] within_channel + refract_viol\n');
        within_channel(channels);
        refract_viol(channels);
        apply_masks_to_times(channels, 'times_dir', tim_work, 'spikes_dir', spk_work, ...
                             'masks', o.masks);
    end

    % ---------- 6. MASKED trial ----------
    if want('masked')
        snapshot(spk_work, tim_work, d_masked, channels, NSx);
        trial_grapes(d_masked, channels, o, root, false);
        trial_sorting_metrics(d_masked, channels, o, root, false);
        if o.do_plots, trial_plot(d_masked, channels, o, root, false, ''); end
    end

    % ---------- 7. rescue ----------
    if do_rescue
        fprintf('[validate] rescue_spikes\n');
        rescue_spikes(channels, 'parallel', true, 'restore', false, ...
                      'folder_name', ['times_val_' ts], 'spikes_folder', spk_work);

        % ---------- 8. RESCUED trial ----------
        snapshot(spk_work, tim_work, d_resc, channels, NSx);
        trial_grapes(d_resc, channels, o, root, true);
        trial_sorting_metrics(d_resc, channels, o, root, true);
        if o.do_plots
            trial_plot(d_resc, channels, o, root, true, 'post');
            trial_plot(d_resc, channels, o, root, true, 'rescued');
        end
    end

    all_tr = struct( ...
        'name', {'nomask','masked','rescued'}, ...
        'dir',  {d_nomask, d_masked, d_resc}, ...
        'rescue', {false, false, true});
    keep = arrayfun(@(t) want(t.name), all_tr);
    trials = all_tr(keep);
    [trials.metrics] = deal('');
    out = struct('out_dir', o.out_dir, 'grapes_tag', gtag, 'trials', trials, 'cmp', struct());

    cmp_dir = fullfile(o.out_dir, 'comparisons');
    if ~exist(cmp_dir,'dir'), mkdir(cmp_dir); end

    % ---------- sorting-quality metrics across trials ----------
    if o.do_sorting_metrics
        try
            out.sorting_metrics = collect_sorting_metrics(trials, channels, cmp_dir);
        catch ME
            warning('validate_mask_effects:sortMetricsFailed', '%s', ME.message);
        end
    end

    if ~o.do_compare || numel(trials) < 2
        if numel(trials) < 2
            fprintf('[validate] compare needs >=2 trials (have %d) - skipping compare stage.\n', numel(trials));
        else
            fprintf('[validate] done (compare stage skipped).\n');
        end
        return;
    end

    % ---------- 9. characterize each trial ----------
    ctag = chan_tag(channels);
    for t = 1:numel(trials)
        fprintf('[validate] characterize_grapes_responses in %s\n', trials(t).name);
        run_in(trials(t).dir, @() characterize_grapes_responses( ...
            'channels2plot', channels, 'stim_list', o.stim_list, 'session_root', root));
        trials(t).metrics = fullfile(trials(t).dir, ...
            sprintf('grapes_response_metrics_%s.mat', ctag));
    end
    out.trials = trials;

    % ---------- 10. compare per (channel, unit) ----------
    for ci = 1:numel(channels)
        ch = channels(ci);
        units = units_for_channel(trials(1).metrics, gtag, ch, o.compare_classes);
        for u = 1:numel(units)
            cn = units{u};
            key = sprintf('ch%d_%s', ch, cn);
            fprintf('[validate] compare %s\n', key);
            try
                c = compare_grapes_response_trials( ...
                    'trials', {trials.metrics}, 'mode', 'trials', 'prefer_rescue', true, ...
                    'grapes_tag', gtag, 'class', cn, 'channel', ch, ...
                    'stim_list', o.stim_list, ...
                    'labels', {trials.name}, ...
                    'reference', trials(1).name, ...
                    'max_plot_stims', o.max_plot_stims, ...
                    'save_mat', true, 'outfile', fullfile(cmp_dir, [key '.mat']), ...
                    'make_plots', true, 'plot_dir', fullfile(cmp_dir, [key '_plots']));
                out.cmp.(matlab.lang.makeValidName(key)) = c;
            catch ME
                warning('validate_mask_effects:cmpFailed', '%s: %s', key, ME.message);
            end
        end
    end

    fprintf('[validate] DONE. Comparisons in %s\n', cmp_dir);
end

% ======================================================================
function stem = grapes_stem_for(use_blanks, circ)
    if use_blanks && circ,     stem = 'grapes_blanks_circ';
    elseif use_blanks,         stem = 'grapes_blanks';
    else,                      stem = 'grapes';
    end
end

% ----------------------------------------------------------------------
function t = chan_tag(channels)
    c = round(double(channels(:)'));
    if isscalar(c)
        t = sprintf('ch%d', c);
    else
        t = ['ch' strjoin(arrayfun(@(x) sprintf('%d', x), c, 'UniformOutput', false), '-')];
    end
end

% ----------------------------------------------------------------------
function copy_channel_files(src_dir, dest_dir, channels, NSx, suffix)
    for ch = channels(:)'
        pos = find(arrayfun(@(x) x.chan_ID == ch, NSx), 1);
        if isempty(pos), warning('chan %d not in NSx', ch); continue; end
        f = fullfile(src_dir, [NSx(pos).output_name suffix]);
        if exist(f, 'file')
            copyfile(f, dest_dir);
        else
            warning('validate_mask_effects: %s not found', f);
        end
    end
end

% ----------------------------------------------------------------------
function snapshot(spk_dir, tim_dir, dest, channels, NSx)
    if ~exist(dest, 'dir'), mkdir(dest); end
    for ch = channels(:)'
        pos = find(arrayfun(@(x) x.chan_ID == ch, NSx), 1);
        if isempty(pos), continue; end
        nm = NSx(pos).output_name;
        sf = fullfile(spk_dir, [nm '_spikes.mat']);
        tf = fullfile(tim_dir, ['times_' nm '.mat']);
        if exist(sf, 'file'), copyfile(sf, dest); end
        if exist(tf, 'file'), copyfile(tf, dest); end
    end
    tpl = fullfile(tim_dir, 'templates_wc_offline.mat');
    if exist(tpl, 'file'), copyfile(tpl, dest); end
end

% ----------------------------------------------------------------------
function trial_grapes(D, channels, o, root, do_rescue)
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

% ----------------------------------------------------------------------
function trial_sorting_metrics(D, channels, o, root, is_rescue)
    if ~o.do_sorting_metrics, return; end
    cd(root);
    try
        compute_metrics_batch(channels, 'times_dir', D, 'spikes_dir', D, ...
                              'save', true, 'parallel', true, 'rescue', is_rescue);
    catch ME
        warning('validate_mask_effects:metricsFailed', '%s: %s', D, ME.message);
    end
end

% ----------------------------------------------------------------------
% collect_sorting_metrics moved to responses/collect_sorting_metrics.m
% (shared with analyze_cluster_membership.m -- same stacking/plot code).

% ----------------------------------------------------------------------
function trial_plot(D, channels, o, root, is_rescue, units)
    cd(root);
    args = {'grapes_offline', true, 'target_dir', D, 'channels2plot', channels, ...
            'use_blanks', o.use_blanks, 'circshiftblanks', o.circshiftblanks, ...
            'stim_list', o.stim_list, 'order_by_rank', false, ...
            'plot_best_stims_only', false, 'copy2miniscrfolder', false, ...
            'is_online', false};
    if is_rescue
        args = [args, {'rescue', true, 'rescue_units', units}];
    end
    try
        plot_grapes_as_online(args{:});
    catch ME
        warning('validate_mask_effects:plotFailed', '%s (%s): %s', D, units, ME.message);
    end
end

% ----------------------------------------------------------------------
function run_in(d, fn)
    old = pwd;
    c = onCleanup(@() cd(old));
    cd(d);
    fn();
end

% ----------------------------------------------------------------------
function units = units_for_channel(metrics_file, gtag, ch, compare_classes)
% mu + the class numbers present for this channel in the nomask metrics.
    units = {'mu'};
    try
        m = load(metrics_file);
        if ~isfield(m, gtag), return; end
        Tt = m.(gtag).metrics;
        sel = Tt.channel_num == ch;
        cl  = unique(cellstr(string(Tt.class(sel))));
        nums = [];
        for i = 1:numel(cl)
            tok = regexp(cl{i}, '^class(\d+)$', 'tokens', 'once');
            if ~isempty(tok), nums(end+1) = str2double(tok{1}); end
        end
        nums = sort(unique(nums));
        if ~(ischar(compare_classes) || isstring(compare_classes))
            nums = intersect(nums, double(compare_classes(:)'));
        end
        for n = nums
            units{end+1} = sprintf('class%d', n);
        end
    catch
    end
end
