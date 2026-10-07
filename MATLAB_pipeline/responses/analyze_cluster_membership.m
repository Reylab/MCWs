function out = analyze_cluster_membership(trials, ref_name, ch, varargin)
% ANALYZE_CLUSTER_MEMBERSHIP  Per-class spike-set membership, fresh
% per-(stimulus,class) response metrics compared PAIRWISE, and mu-level
% window tracing, across "trials" that share one channel.
%
% GRAPES_BLANKS ONLY: every trial's grapes file must be blanks-flavour
% (filename contains 'blank', same convention characterize_grapes_responses
% uses) -- errors otherwise. No plain-vs-blanks window mismatch is tolerated
% here; that's a different problem, handled elsewhere.
%
% USAGE
%   trials(1) = struct('name','handselect', 'grapes_file',gf1, 'times_file',tf1, 'spikes_file','');
%   trials(2) = struct('name','masked',      'grapes_file',gf2, 'times_file',tf2, 'spikes_file',sf2);
%   trials(3) = struct('name','rescue_sd1',  'grapes_file',gf3, 'times_file',tf3, 'spikes_file',sf3);
%   ...
%   out = analyze_cluster_membership(trials, 'handselect', 337, 'root', root)
%
% INPUTS
%   trials     struct array, one per trial:
%                .name         short label
%                .grapes_file  grapes_blanks*.mat for this trial (REQUIRED)
%                .times_file   times_<nm>.mat (cluster_class/spikes) -- needed
%                              for the per-class membership piece
%                .spikes_file  <nm>_spikes.mat (spikes_all/index_all/masks) --
%                              needed for the mu-window "trace the missing
%                              ones" piece; optional otherwise
%   ref_name   trials(i).name to treat as ground truth for the *-only sets
%              and left/right/missing tracing. Response-metric plots still
%              cover every pair, not just vs ref.
%   ch         channel number
%
%   Name/value:
%     'root'            ''     session root (stimulus.mat/finalevents.mat/
%                                experiment_properties_online3.mat) -- needed
%                                to build the onset map
%     'gt_classes'       'all'  which reference classes to analyze
%     'match_tol_ms'      0.5   one-to-one spike matching tolerance
%     'window_tol_ms'    20     slack added to the response window when
%                                classifying a mu spike as inside/left/right
%                                of it (a latency margin, not a hard cutoff)
%     'reject_sdnum_xref'  3    radius (in a class's own std) used when
%                                tracing missing mu spikes against a trial's
%                                own clustering, to say "looks like class k"
%     'out_dir'           ''    default: alongside trials(ref).grapes_file
%
% OUTPUT: out struct (membership/, response/, mu_window/) + files under
% out_dir with matching subfolder names.

    p = inputParser;
    addParameter(p, 'root', '');
    addParameter(p, 'gt_classes', 'all');
    addParameter(p, 'match_tol_ms', 0.5);
    addParameter(p, 'window_tol_ms', 20);
    addParameter(p, 'reject_sdnum_xref', 3);
    addParameter(p, 'stim_list', []);   % ORDERED stim numbers for the raster stage, your
                                        % order verbatim -- 20-40 at a time. Empty falls back
                                        % to every stim present, numeric order (with a warning).
    addParameter(p, 'masked_spikes_file', '');  % pre-rescue masked arm's <nm>_spikes.mat
                                        % (quarantine_properties + index_all) -- OPTIONAL, but
                                        % needed for the within_channel-check rasters (1b) AND the
                                        % rejected/rescued response-assessment (2d). Empty skips
                                        % both with a warning.
    addParameter(p, 'resp_z_thr', 3);  % |zscore| cutoff (on top of the pipeline's own
                                        % min_spk_test & p_test & good_lat gate) for calling a
                                        % (trial,gt_class,stim) responsive -- same default as
                                        % validate_recluster_vs_handselect.
    addParameter(p, 'out_dir', '');
    parse(p, varargin{:});
    o = p.Results;
    addpath(genpath('/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline'));

    % dedupe stim_list ONCE, here, so every consumer below (rasters, response
    % metrics, pairwise plots, ...) sees a clean list -- a repeated stim
    % number produces two rows per trial for it, and any code that expects
    % "the row for stim s" (stimA==s style indexing) breaks on the second
    % one. Confirmed crash: pairwise_metric_plots death on a stim_list with
    % a duplicate. First occurrence kept, order otherwise preserved.
    if ~isempty(o.stim_list)
        [uniq_stims, first_idx] = unique(o.stim_list(:)', 'stable');
        if numel(uniq_stims) < numel(o.stim_list)
            dupes = o.stim_list(setdiff(1:numel(o.stim_list), first_idx));
            warning('acm:dupStimList', 'stim_list has duplicate stim number(s) %s -- keeping first occurrence only.', ...
                mat2str(unique(dupes)));
            o.stim_list = uniq_stims;
        end
    end

    % ---- gate: blanks flavour only -----------------------------------------
    for i = 1:numel(trials)
        [~, gbase, gext] = fileparts(trials(i).grapes_file);
        assert(contains([gbase gext], 'blank', 'IgnoreCase', true), ...
            'analyze_cluster_membership: trials(%d) ("%s") grapes_file "%s%s" is not a blanks raster. ONLY grapes_blanks rasters are accepted here.', ...
            i, trials(i).name, gbase, gext);
    end
    ref_i = find(strcmp({trials.name}, ref_name), 1);
    assert(~isempty(ref_i), 'ref_name "%s" not found in trials.', ref_name);

    if isempty(o.out_dir)
        [d0, ~, ~] = fileparts(trials(ref_i).grapes_file);
        o.out_dir = fullfile(d0, 'membership_analysis');
    end
    mdir = fullfile(o.out_dir, 'membership'); rdir = fullfile(o.out_dir, 'response'); wdir = fullfile(o.out_dir, 'mu_window');
    rasdir = fullfile(o.out_dir, 'rasters');
    cellfun(@(d) ~exist(d,'dir') && mkdir(d), {mdir, rdir, wdir, rasdir});

    assert(~isempty(o.root), 'analyze_cluster_membership: ''root'' is required (for the onset map).');
    omap = build_onset_map(o.root);

    % ---- reconstruct every trial's per-class + mu post-onset spikes -------
    for i = 1:numel(trials)
        trials(i).cc = reconstruct_class_cc(trials(i).grapes_file, ch, omap);   % [class time_ms]
        trials(i).mu = reconstruct_mu_times(trials(i).grapes_file, ch, omap);   % time_ms
        trials(i).raw = load(trials(i).grapes_file);                            % keep raw rasters for metrics
    end
    ref = trials(ref_i);
    gt_classes_all = unique(ref.cc(:,1))';
    if ischar(o.gt_classes) && strcmp(o.gt_classes,'all')
        gt_classes = gt_classes_all;
    else
        gt_classes = intersect(o.gt_classes, gt_classes_all);
    end
    fprintf('[acm] ch%d ref="%s" classes: %s\n', ch, ref_name, mat2str(gt_classes));
    if isempty(gt_classes)
        error('acm:emptyGtClasses', ['ch%d ref="%s": gt_classes is empty -- ref.cc reconstructed 0 ' ...
            'post-onset spikes for any class. This usually means build_onset_map found no matching ' ...
            'stimulus onsets (check stimulus.mat''s st(j).ID against the raster''s positional stim ' ...
            'numbering), not that the ground-truth times file is actually empty.'], ch, ref_name);
    end

    % ======================= 0b. class matching (where do spikes go) =======
    % Real match_cc/class_match_figure (from validate_recluster_vs_handselect.m,
    % extracted to shared files) -- NOT reimplemented. For each trial: where
    % each GT class's spikes end up (arm classK / arm class0 / unmatched) and
    % what the matched arm cluster is made of (this GT class / other GT / GT
    % class0 / new spikes with no GT counterpart). Filtered to gt_classes
    % (the one class this run is scoped to) before saving/plotting. Lives in
    % mdir, alongside the venn diagrams.
    % Same full-vs-windowed rule as membership below: per (gt_class,trial),
    % use the times file's full whole-session classes ONLY when BOTH GT's
    % and the matched arm's times-vs-grapes counts agree; otherwise fall
    % back to the windowed reconstruction for BOTH sides (never mixed).
    match_tbl = table(); detail_tbl = table(); comp_tbl = table();
    for i = 1:numel(trials)
        if i == ref_i, continue; end
        for gi = 1:numel(gt_classes)
            gtc = gt_classes(gi);
            arm_class = match_arm_class(ref.cc, gtc, trials(i).cc);
            if isnan(arm_class), continue; end
            [gt_ok,~,~]  = check_class_count_agreement(ref.times_file, ref.grapes_file, ch, gtc);
            [arm_ok,~,~] = check_class_count_agreement(trials(i).times_file, trials(i).grapes_file, ch, arm_class);
            if gt_ok && arm_ok
                ccg = full_cc_from_times(ref.times_file);
                cca = full_cc_from_times(trials(i).times_file);
            else
                ccg = ref.cc;
                cca = trials(i).cc;
            end
            [T, DET, COMP] = match_cc(ccg, cca, ch, o.match_tol_ms, 0.5);
            keep_T = T.gt_class == gtc; keep_D = DET.gt_class == gtc; keep_C = COMP.gt_class == gtc;
            T = T(keep_T,:); DET = DET(keep_D,:); COMP = COMP(keep_C,:);
            T.arm    = repmat(string(trials(i).name), height(T), 1);
            DET.arm  = repmat(string(trials(i).name), height(DET), 1);
            COMP.arm = repmat(string(trials(i).name), height(COMP), 1);
            match_tbl  = [match_tbl;  T];
            detail_tbl = [detail_tbl; DET];
            comp_tbl   = [comp_tbl;   COMP];
        end
    end
    if ~isempty(match_tbl)
        match_tbl = movevars(match_tbl, {'arm','channel_id','gt_class'}, 'Before', 1);
        writetable(match_tbl, fullfile(mdir, sprintf('class_match_ch%d.csv', ch)));
        writetable(movevars(detail_tbl, 'arm', 'Before', 1), ...
            fullfile(mdir, sprintf('class_match_where_gt_spikes_go_ch%d.csv', ch)));
        writetable(movevars(comp_tbl, 'arm', 'Before', 1), ...
            fullfile(mdir, sprintf('class_match_arm_cluster_composition_ch%d.csv', ch)));
        other_names = {trials(setdiff(1:numel(trials), ref_i)).name};
        try
            class_match_figure(detail_tbl, comp_tbl, ch, other_names, mdir);
        catch ME
            warning('acm:classMatchFigure', '%s', ME.message);
        end
        fprintf('[acm] class matching -> %s\n', mdir);
    end

    % ======================= 1. per-class membership ========================
    % GT's own raster (real file, every class, no synthesis) is IDENTICAL
    % for every (gt_class,trial) iteration below -- same file, same
    % stim_order -- so build it ONCE here and copy the result into each
    % trial's gt/ folder, instead of re-running plot_grapes_as_online
    % (~15-25s, re-caching the whole stimulus image set) once per trial.
    gt_raster_src = '';
    if ~isempty(o.stim_list)
        gt_raster_src = fullfile(rasdir, '_gt_shared');
        run_plot_grapes_as_online(ref.grapes_file, false, ch, o.stim_list(:)', gt_raster_src);
        if ~exist(gt_raster_src, 'dir'), gt_raster_src = ''; end
    end

    membership = struct('gt_class',{},'trial',{},'arm_class',{},'scope',{},'n_gt',{},'n_arm',{}, ...
                        'n_matched',{},'n_gt_only',{},'n_arm_only',{},'pct_gt_lost',{},'pct_arm_new',{}, ...
                        'n_gt_full',{},'n_gt_windowed',{},'n_arm_full',{},'n_arm_windowed',{});
    for gi = 1:numel(gt_classes)
        gtc = gt_classes(gi);
        gt_t_win = ref.cc(ref.cc(:,1)==gtc, 2);
        [gt_match_ok, gt_n_times, ~] = check_class_count_agreement(ref.times_file, ref.grapes_file, ch, gtc);
        for i = 1:numel(trials)
            if i == ref_i, continue; end
            % identify the matched arm class off the windowed reconstruction,
            % always -- this is a spike-time-overlap lookup, not a count, so
            % the times-vs-grapes agreement question doesn't apply to it.
            [arm_class, arm_t_win] = match_arm_class(ref.cc, gtc, trials(i).cc);
            if isnan(arm_class)
                fprintf('  [acm] gt class %d: no matching class in trial "%s" (skipped)\n', gtc, trials(i).name);
                continue
            end
            [arm_match_ok, arm_n_times, ~] = check_class_count_agreement(trials(i).times_file, ...
                trials(i).grapes_file, ch, arm_class);
            % full (times-file) counts used only when BOTH sides' times file
            % agrees with its own grapes details for this class -- never mix
            % full vs windowed within one comparison.
            use_full = gt_match_ok && arm_match_ok;
            if use_full
                gt_t  = full_class_times_from_times_file(ref.times_file, gtc);
                arm_t = full_class_times_from_times_file(trials(i).times_file, arm_class);
                scope = 'full';
            else
                gt_t  = gt_t_win;
                arm_t = arm_t_win;
                scope = 'windowed';
            end
            [matched_gt, matched_arm, gt_only, arm_only] = membership_sets(gt_t, arm_t, o.match_tol_ms);
            n_gt = numel(gt_t); n_arm = numel(arm_t);
            row = struct('gt_class',gtc, 'trial',trials(i).name, 'arm_class',arm_class, 'scope',string(scope), ...
                'n_gt',n_gt, 'n_arm',n_arm, 'n_matched',numel(matched_gt), ...
                'n_gt_only',numel(gt_only), 'n_arm_only',numel(arm_only), ...
                'pct_gt_lost', 100*numel(gt_only)/max(n_gt,1), ...
                'pct_arm_new', 100*numel(arm_only)/max(n_arm,1), ...
                'n_gt_full', gt_n_times, 'n_gt_windowed', numel(gt_t_win), ...
                'n_arm_full', arm_n_times, 'n_arm_windowed', numel(arm_t_win));
            membership(end+1) = row;
            % mismatch -> venn is sized/drawn off the windowed numbers always
            % (gt_t/arm_t already ARE windowed in that case); the full
            % numbers are passed through only as text annotation, never used
            % for sizing -- both scopes visible, only one ever drives the plot.
            plot_membership_set(gtc, trials(i).name, arm_class, gt_t, arm_t, ...
                matched_gt, gt_only, arm_only, mdir, ch, scope, gt_n_times, arm_n_times);
            % ---- rasters, LAST stage: arm / gt / matched / gt_only / arm_only,
            % one folder per trial (the sd# is the folder split), evaluated for
            % this one GT class only -- reuses the exact same matched/gt_only/
            % arm_only spike sets just computed, no recomputation.
            %
            % arm & gt: their grapes files ALREADY EXIST (built by
            % build_masked_arm/build_rescue_sd_trials, and the hand-select
            % ground truth) -- nothing to synthesize, plot_grapes_as_online
            % runs directly on the real file.
            % matched/gt_only/arm_only: these spike-time sets exist nowhere
            % in any real grapes file (they're derived by the venn logic
            % above), so a minimal single-class grapes_blanks.mat is built
            % for each and plot_grapes_as_online runs on THAT.
            trial_ras_dir = fullfile(rasdir, sprintf('gt%d', gtc), trials(i).name);
            if ~exist(trial_ras_dir, 'dir'), mkdir(trial_ras_dir); end
            gt_detail  = reconstruct_class_detail(ref.grapes_file, ch, gtc, omap);
            arm_detail = reconstruct_class_detail(trials(i).grapes_file, ch, arm_class, omap);
            if ~isempty(o.stim_list)
                stim_order = o.stim_list(:)';   % your order, verbatim -- not sorted
            else
                stim_order = unique([gt_detail.snum, arm_detail.snum], 'stable');
                warning('acm:noStimList', ['no ''stim_list'' given -- rasters falling back to every ' ...
                    'stim present in first-seen order. Pass an explicit ordered 20-40 stim ''stim_list''.']);
            end
            % arm & gt rasters: plot every class (real file, as-is) -- the
            % raster is a visual/qualitative aid, fine to show full context.
            % Only the metric ANALYSIS below (response/pairwise/change
            % summary) is scoped to gt_classes (this one class) -- that's
            % where "per stim per trial, one comparison" actually lives.
            is_rescue_arm = contains(trials(i).grapes_file, '_rescue');
            run_plot_grapes_as_online(trials(i).grapes_file, is_rescue_arm, ch, stim_order, ...
                fullfile(trial_ras_dir, 'arm'));
            gt_dest = fullfile(trial_ras_dir, 'gt');
            if ~isempty(gt_raster_src)
                % built once, above the gt_classes loop -- just copy it in.
                if ~exist(gt_dest,'dir'), mkdir(gt_dest); end
                copyfile(fullfile(gt_raster_src,'*'), gt_dest);
            else
                run_plot_grapes_as_online(ref.grapes_file, false, ch, stim_order, gt_dest);
            end

            % Pass the TRUE matched/gt_only/arm_only sets straight in (not
            % windowed-intersected against gt_detail/arm_detail first) --
            % update_grapes_blanks(domu=true) sets details.mu = numel(input)
            % regardless of how many of those actually land in a raster row
            % (offline/is_online=false path -- confirmed against
            % update_grapes_blanks.m), same mechanism every real arm/gt
            % raster already relies on for its own "(N spks)" title. So the
            % title shows the SAME scope-correct number already in the
            % venn/CSV (numel(matched_gt)==numel(matched_arm) always, by
            % construction of membership_sets' one-to-one pairing), while the
            % actual drawn ticks are still only whichever of those fall in a
            % stimulus window -- a raster can't draw a spike with no
            % relative-onset-time, same limitation real rasters have too.
            derived = { ...
                'matched',  matched_arm; ...
                'gt_only',  gt_only; ...
                'arm_only', arm_only };
            for si = 1:size(derived,1)
                synth_dir = fullfile(trial_ras_dir, [derived{si,1} '_synth']);
                gf = synth_grapes_for_subset(derived{si,2}, ch, o.root, ...
                    ref.grapes_file, synth_dir, nm_for_ch(o.root, ch), false);
                if isempty(gf)
                    fprintf('  [acm] gt%d %s %s: 0 spikes, no raster to plot.\n', ...
                        gtc, trials(i).name, derived{si,1});
                    continue;
                end
                run_plot_grapes_as_online(gf, false, ch, stim_order, ...
                    fullfile(trial_ras_dir, derived{si,1}));
            end
        end
    end
    if ~isempty(gt_raster_src) && exist(gt_raster_src, 'dir')
        rmdir(gt_raster_src, 's');   % staging copy only, already fanned out above
    end
    T_mem = struct2table(membership);
    if ~isempty(T_mem), writetable(T_mem, fullfile(mdir, sprintf('membership_ch%d.csv', ch))); end
    fprintf('[acm] membership: %d rows -> %s\n', height(T_mem), mdir);

    % ======================= 1b. within_channel-check rasters ================
    % Per within_channel check (structural/amp/width/prom_ratio/multipeak):
    % a mu raster of what that check FLAGS (quarantines) -- the population
    % within_channel throws away, straight from the masked arm's own
    % quarantine_properties/index_all, same fields mask_audit.m reads, no
    % re-running within_channel. Then, per sdnum rescue trial: a mu raster
    % of what THAT sdnum rescues back out of that check's flagged
    % population (rescue_mask, aligned to the same index_all, same as
    % mask_audit's pct_rescued), plus a per-class raster restricted to the
    % matched arm_class for THIS gt_class only (not every class -- same
    % one-class scope as everything else here).
    if isempty(o.masked_spikes_file) || ~exist(o.masked_spikes_file, 'file')
        warning('acm:noWcRasters', ['no ''masked_spikes_file'' given (or not found) -- skipping ' ...
            'within_channel-check rasters.']);
    else
        MS = load(o.masked_spikes_file);
        if isfield(MS,'index_all'), idx_all = double(MS.index_all(:)); else, idx_all = double(MS.index(:)); end
        if ~isfield(MS, 'quarantine_properties')
            warning('acm:noWcRasters', 'masked_spikes_file has no quarantine_properties -- skipping within_channel-check rasters.');
        else
            qp = MS.quarantine_properties;
            wc_checks = {'structural','amp','width','prom_ratio','multipeak'};
            wc_fld = strcat('fail_', wc_checks);
            have = cellfun(@(f) isfield(qp,f), wc_fld);
            wc_checks = wc_checks(have); wc_fld = wc_fld(have);
            Fwc = false(numel(idx_all), numel(wc_checks));
            for ci = 1:numel(wc_checks), Fwc(:,ci) = logical(qp.(wc_fld{ci})(:)); end
            nm_ch = nm_for_ch(o.root, ch);
            wc_stim_order = o.stim_list(:)';

            % 2d. rejected/rescued response assessment (real
            % characterize_grapes_responses on the SAME synth files just
            % built for the rasters above -- does the population a check
            % discards actually carry a stimulus response (bad -- losing
            % signal) or not (fine, safe to discard); does rescue recover
            % a real response) -- same responsive gate as everywhere else
            % (min_spk_test & p_test & good_lat & |zscore|>=resp_z_thr).
            assess_rows = struct('check',{},'subset',{},'trial',{},'n_spikes',{},'n_responsive',{}, ...
                'frac_responsive',{},'mean_zscore',{},'max_zscore',{},'mean_resp_delta_fr_hz',{});

            for gi = 1:numel(gt_classes)
                gtc = gt_classes(gi);
                wcdir = fullfile(rasdir, sprintf('gt%d', gtc), 'wc_checks');
                for ci = 1:numel(wc_checks)
                    ckdir = fullfile(wcdir, wc_checks{ci});
                    flagged_abs = idx_all(Fwc(:,ci));
                    if ~isempty(flagged_abs)
                        gf = synth_grapes_for_subset(flagged_abs, ch, o.root, ref.grapes_file, ...
                            fullfile(ckdir, 'flagged_mu_synth'), nm_ch, false);
                        run_plot_grapes_as_online(gf, false, ch, wc_stim_order, fullfile(ckdir, 'flagged_mu'));
                        assess_rows(end+1) = summarize_wc_response(gf, ch, wc_stim_order, o.root, ...
                            wc_checks{ci}, 'flagged', '', numel(flagged_abs), o.resp_z_thr);
                    end
                    for i = 1:numel(trials)
                        if i == ref_i, continue; end
                        tf = trials(i).times_file;
                        if isempty(tf) || ~exist(tf, 'file'), continue; end
                        RT = load(tf, 'rescue_mask', 'cluster_class');
                        if ~isfield(RT,'rescue_mask') || numel(RT.rescue_mask) ~= numel(idx_all), continue; end
                        rmask = logical(RT.rescue_mask(:));
                        resc_abs = idx_all(Fwc(:,ci) & rmask);
                        if isempty(resc_abs), continue; end
                        sd_dir = fullfile(ckdir, trials(i).name);
                        gf = synth_grapes_for_subset(resc_abs, ch, o.root, ref.grapes_file, ...
                            fullfile(sd_dir, 'rescued_mu_synth'), nm_ch, false);
                        run_plot_grapes_as_online(gf, false, ch, wc_stim_order, fullfile(sd_dir, 'rescued_mu'));
                        assess_rows(end+1) = summarize_wc_response(gf, ch, wc_stim_order, o.root, ...
                            wc_checks{ci}, 'rescued', trials(i).name, numel(resc_abs), o.resp_z_thr);

                        acls = match_arm_class(ref.cc, gtc, trials(i).cc);
                        if ~isnan(acls) && isfield(RT, 'cluster_class')
                            gm = nearest_match(resc_abs, RT.cluster_class(:,2), o.match_tol_ms);
                            keep = false(size(gm)); ok = gm > 0;
                            keep(ok) = RT.cluster_class(gm(ok), 1) == acls;
                            class_abs = resc_abs(keep);
                            if ~isempty(class_abs)
                                gf = synth_grapes_for_subset(class_abs, ch, o.root, ref.grapes_file, ...
                                    fullfile(sd_dir, sprintf('rescued_class%d_synth', acls)), nm_ch, false);
                                run_plot_grapes_as_online(gf, false, ch, wc_stim_order, ...
                                    fullfile(sd_dir, sprintf('rescued_class%d', acls)));
                                assess_rows(end+1) = summarize_wc_response(gf, ch, wc_stim_order, o.root, ...
                                    wc_checks{ci}, sprintf('rescued_class%d',acls), trials(i).name, ...
                                    numel(class_abs), o.resp_z_thr);
                            end
                        end
                    end
                end
            end
            if ~isempty(assess_rows)
                writetable(struct2table(assess_rows), fullfile(mdir, sprintf('assess_wc_checks_ch%d.csv', ch)));
            end
            if isempty(gt_classes)
                fprintf('[acm] within_channel-check rasters + response assessment: 0 gt_classes -- nothing written under %s\n', ...
                    rasdir);
            else
                fprintf('[acm] within_channel-check rasters + response assessment -> %s\n', ...
                    fullfile(rasdir, sprintf('gt%d', gt_classes(1)), 'wc_checks'));
            end
        end
    end

    % ======================= 2. response metrics, fresh, pairwise ===========
    % Response metrics now come straight from the REAL characterize_grapes_
    % responses (-> create_responses_data_parallel, the exact function
    % plot_grapes_as_online/do_structure_* use). The previous local
    % compute_stim_class_metrics reimplemented zscore/p_value_sign/onset/
    % duration/peak_ifr with different math (per-stim-only baseline stats +
    % a fixed window + a home-made IFR, vs the real per-class population
    % baseline pooled across the scored stim set + an adaptive per-stim
    % latency window off the real IFR calculator) -- confirmed by reading
    % create_responses_data_parallel.m; only resp_delta_fr_hz/auroc/
    % first_spk_lat_med_ms (add-ons characterize_grapes_responses itself
    % computes, not part of the core per-class pipeline metrics) matched.
    % Removed rather than patched -- run the real thing instead.
    if ~isempty(o.stim_list)
        all_stims = o.stim_list(:)';   % YOUR list, verbatim order -- not sorted, not auto-derived
    else
        all_stims = [];
        for i=1:numel(trials)
            rr = trials(i).raw.rasters.(sprintf('chan%d',ch));
            all_stims = union(all_stims, find(~cellfun(@isempty, rr.mu.stim)), 'stable');
        end
        warning('acm:noStimList', ['no ''stim_list'' given -- response metrics falling back to every ' ...
            'stim present (%d stims). Pass an explicit ordered ''stim_list'' to restrict/order this.'], ...
            numel(all_stims));
    end

    % NOTE: characterize_grapes_responses pools baseline mu/std ACROSS every
    % stimulus it scores (that's how the real zscore/p_value_sign are
    % defined -- see create_responses_data_parallel). Scoring only
    % all_stims (your 20-40) means that population is your subset, not the
    % full experiment -- matches "stim_list controls the comparison", but
    % the zscore values will differ from a full-experiment online run.
    trial_tbl = containers.Map('KeyType','char','ValueType','any');
    for i = 1:numel(trials)
        try
            trial_tbl(trials(i).name) = run_characterize_grapes(trials(i).grapes_file, ch, all_stims, o.root);
        catch ME
            warning('acm:charFailed', 'trial "%s": characterize_grapes_responses failed: %s', ...
                trials(i).name, ME.message);
        end
    end

    metrics = struct('trial',{},'gt_class',{},'arm_class',{},'stim',{},'m',{});
    arm_class_for = containers.Map('KeyType','double','ValueType','any');  % gt_class -> trial_name -> arm_class
    for gi = 1:numel(gt_classes)
        gtc = gt_classes(gi);
        armmap = containers.Map('KeyType','char','ValueType','double');
        armmap(ref_name) = gtc;
        for i = 1:numel(trials)
            if i==ref_i, continue; end
            ac = match_arm_class(ref.cc, gtc, trials(i).cc);
            armmap(trials(i).name) = ac;
        end
        arm_class_for(gtc) = armmap;
        for i = 1:numel(trials)
            ac = armmap(trials(i).name);
            if isnan(ac) || ~isKey(trial_tbl, trials(i).name), continue; end
            Tt = trial_tbl(trials(i).name);
            if isempty(Tt), continue; end
            cf = sprintf('class%d', ac);
            sel_cls = strcmp(Tt.class, cf);
            for si = 1:numel(all_stims)
                snum = all_stims(si);
                r = Tt(sel_cls & Tt.stim_number==snum, :);
                if isempty(r), continue; end
                r = r(1,:);
                % full real column set (not just the original 9) -- everything
                % characterize_grapes_responses computes, incl. the pipeline's
                % own responsive gate (min_spk_test/p_test/good_lat), used by
                % compute_responsive_summary below and kept here (not in any
                % PLOTTED flds list -- they're booleans) rather than recomputed.
                m = struct('zscore', r.zscore, 'mu_base', r.mu_base, 'std_base', r.std_base, ...
                    'median_post', r.median_post, 'p_value_sign', r.p_value_sign, 'p_ranksum', r.p_ranksum, ...
                    'auroc', r.auroc, 'resp_delta_fr_hz', r.resp_delta_fr_hz, ...
                    'mean_fr_base_hz', r.mean_fr_base_hz, 'mean_fr_post_hz', r.mean_fr_post_hz, ...
                    'spk_post_mean', r.spk_post_mean, 'spk_post_std', r.spk_post_std, ...
                    'frac_active_trials', r.frac_active_trials, 'peak_ifr', r.peak_ifr, ...
                    'peak_ifr_lat', r.peak_ifr_lat, 'onset_ms', r.onset, 'duration_ms', r.dura, ...
                    'tons_ms', r.tons, 'ntrials', r.ntrials, 'ifr_thr', r.ifr_thr, ...
                    'first_spk_lat_med_ms', r.first_spk_lat_med, ...
                    'min_spk_test', r.min_spk_test, 'p_test', r.p_test, 'good_lat', r.good_lat);
                metrics(end+1) = struct('trial',trials(i).name,'gt_class',gtc,'arm_class',ac,'stim',snum,'m',m);
            end
        end
    end
    fprintf('[acm] response metrics: %d (trial,class,stim) rows\n', numel(metrics));
    pairwise_metric_plots(metrics, {trials.name}, ref_name, gt_classes, rdir, ch);
    change_vs_ref_summary(metrics, T_mem, {trials.name}, ref_name, gt_classes, rdir, ch);
    % "All trials, all stimuli, one gt_class, at once": line per trial, x =
    % stim (your whole list, your order). Two earlier candidate views
    % (per-stimulus bar panels; a stim x trial heatmap) were tried and
    % dropped -- same numbers, more files, less useful than this one.
    plot_metric_trends(metrics, gt_classes, all_stims, {trials.name}, ref_name, rdir, ch);
    % Responsive-stimulus summary: real pipeline gate (min_spk_test & p_test &
    % good_lat) AND |zscore|>=resp_z_thr, same as validate_recluster_vs_
    % handselect's build_compare/compare_plots -- n_responsive/frac_responsive
    % per trial, PLUS n_gained/n_lost/jaccard_resp vs ref (the actual payoff
    % metric: did rescue restore stimuli masking made non-responsive).
    compute_responsive_summary(metrics, gt_classes, {trials.name}, ref_name, o.resp_z_thr, rdir, ch);

    % GT-vs-arm mean waveform overlay, per gt_class -- visually confirms the
    % matched arm class is really the same cell, not just "close by recall".
    % Real waveforms: GT's matched by spike time against its own no-mask
    % spikes file (trials(ref_i).spikes_file); each arm's read straight off
    % its own times_file's spikes/cluster_class (rescue writes real
    % waveforms there already -- no re-matching needed for the arms).
    for gi = 1:numel(gt_classes)
        gtc = gt_classes(gi);
        waveform_overlay_plot(ch, gtc, ref, trials, ref_i, arm_class_for(gtc), ...
            trials(ref_i).spikes_file, o.match_tol_ms, mdir);
    end

    % ======================= 2c. spike-quality metrics (compute_metrics_batch) ==
    % SEPARATE from the response metrics above: SNR/ISI/isolation/l_ratio/
    % d_prime/firing_rate/num_spikes AND the actual per-cluster diagnostic
    % plots (waveforms/ISI/correlogram -- compute_cluster_metrics's own
    % report, saved automatically because 'save',true is passed below),
    % straight from the real compute_metrics_batch, run once per trial on
    % its own times_/spikes_ files. The cross-trial stacking + comparison
    % plot reuses collect_sorting_metrics.m verbatim -- the SAME function
    % validate_mask_effects.m calls -- filtered to the one matched
    % arm_class per trial (so "line per cluster" collapses to one line =
    % the tracked class, across trials whose arm_class numbering differs).
    % Own folder (spike_metrics/), never mixed into response/.
    smdir = fullfile(mdir, 'spike_metrics');
    if ~exist(smdir,'dir'), mkdir(smdir); end
    trials4sm = struct('name',{},'dir',{},'rescue',{});
    for i = 1:numel(trials)
        tname = trials(i).name;
        tf = trials(i).times_file; sf = trials(i).spikes_file;
        if isempty(tf) || isempty(sf) || ~exist(tf,'file') || ~exist(sf,'file')
            warning('acm:noSpikeMetrics', ['trial "%s": missing times_file/spikes_file -- ' ...
                'skipping spike-quality metrics for this trial.'], tname);
            continue;
        end
        D = fileparts(tf);
        is_resc = contains(trials(i).grapes_file, '_rescue');
        try
            % 'save',true -> also writes the real per-cluster waveform/ISI/
            % correlogram report pages into D/metrics[ rescue]/, same as
            % validate_mask_effects's trial_sorting_metrics.
            compute_metrics_batch('all', 'times_dir', D, 'spikes_dir', fileparts(sf), ...
                'save', true, 'parallel', false, 'rescue', is_resc);
            trials4sm(end+1) = struct('name', tname, 'dir', D, 'rescue', is_resc);
        catch ME
            warning('acm:spikeMetricsFailed', 'trial "%s": compute_metrics_batch: %s', tname, ME.message);
        end
    end

    spike_metrics = struct('trial',{},'gt_class',{},'arm_class',{},'row',{});
    for gi = 1:numel(gt_classes)
        gtc = gt_classes(gi);
        armmap = arm_class_for(gtc);
        cf = containers.Map('KeyType','char','ValueType','double');
        for i = 1:numel(trials4sm)
            tname = trials4sm(i).name;
            if isKey(armmap, tname) && ~isnan(armmap(tname))
                cf(tname) = armmap(tname);
            end
        end
        if ~isempty(cf)
            gdir = fullfile(smdir, sprintf('gt%d', gtc));
            if ~exist(gdir,'dir'), mkdir(gdir); end
            % copy every trial's REAL per-cluster waveform/ISI/correlogram
            % report (compute_cluster_metrics's make_cluster_report, already
            % saved by the compute_metrics_batch call above into each
            % trial's own <dir>/metrics[ rescue]/) in here too -- ALL
            % trials, handselect (ground truth) included, not just the
            % arms, so the actual spike plots sit with the summary chart
            % and the venn diagrams, same as validate_mask_effects keeps
            % everything under one out_dir.
            for i = 1:numel(trials4sm)
                mdir_i = fullfile(trials4sm(i).dir, ['metrics' repmat(' rescue', 1, trials4sm(i).rescue)]);
                dpng = dir(fullfile(mdir_i, '*_plots*.png'));
                for k = 1:numel(dpng)
                    copyfile(fullfile(dpng(k).folder, dpng(k).name), ...
                        fullfile(gdir, sprintf('%s_%s', trials4sm(i).name, dpng(k).name)));
                end
            end
            file_prefix = sprintf('spike_metrics_ch%d_gt%d', ch, gtc);
            ST = collect_sorting_metrics(trials4sm, ch, gdir, 'cluster_filter', cf, ...
                'file_prefix', file_prefix, ...
                'title_suffix', sprintf('  (gt-class%d, matched arm_class per trial)', gtc));
            if istable(ST) && ~isempty(ST)
                % num_spikes here is compute_metrics_batch's raw whole-session
                % count (it reads times_file's cluster_class directly, same
                % rule as everywhere else -- never trust it alone). Add the
                % windowed companion count + the agreement flag so a reader
                % sees, per row, whether num_spikes is trustworthy as full-
                % scope or should be read as the windowed number instead.
                n_win = nan(height(ST),1); times_match = false(height(ST),1);
                for r = 1:height(ST)
                    tname = char(ST.trial(r)); acl = ST.matched_cluster_id(r);
                    [tf_r, gf_r, cc_r] = trial_files_for_name(tname, ref, ref_name, trials);
                    if isempty(tf_r), continue; end
                    [ok_r,~,~] = check_class_count_agreement(tf_r, gf_r, ch, acl);
                    times_match(r) = ok_r;
                    n_win(r) = nnz(cc_r(:,1)==acl);
                end
                ST.num_spikes_windowed = n_win;
                ST.times_grapes_match  = times_match;
                writetable(ST, fullfile(gdir, [file_prefix '.csv']));
                for r = 1:height(ST)
                    spike_metrics(end+1) = struct('trial', char(ST.trial(r)), 'gt_class', gtc, ...
                        'arm_class', ST.matched_cluster_id(r), 'row', table2struct(ST(r,:)));
                end
            end
        end
    end
    fprintf('[acm] spike-quality metrics: %d (trial,gt_class) rows -> %s\n', numel(spike_metrics), smdir);

    % ======================= 3. mu-window tracing ============================
    % Only the response window itself is needed here (in/left/right/missing
    % classification against tmin_median..tmax_median +/- window_tol_ms) --
    % same blanks-flavour constants characterize_grapes_responses uses
    % (default_resp_conf(is_blanks=true), grapes_blanks-only per the gate above).
    resp_conf = struct('tmin_median',50,'tmax_median',550,'tmin_base',-550,'tmax_base',-50);
    mu_trace_all = struct('trial',{},'n_ref',{},'n_matched',{},'n_left',{},'n_right',{},'n_missing',{}, ...
                          'missing_status',{});
    for i = 1:numel(trials)
        if i == ref_i, continue; end
        tr = mu_window_trace(ref.raw, trials(i).raw, ch, omap, o.match_tol_ms, o.window_tol_ms, ...
                             resp_conf, trials(i), o.reject_sdnum_xref, wdir, trials(i).name);
        mu_trace_all(end+1) = tr;
    end

    % One HTML page indexing everything above -- the fix for "too many
    % subfolders": every table + figure for this run, per gt_class, in one
    % document instead of folder-spelunking. Adapted from
    % validate_recluster_vs_handselect's write_report (same idea: match +
    % response + quality tables, the figures, a stim x trial grid), trimmed
    % to what this tool actually builds (no dashboard/feature_space/
    % cluster_comparison -- those weren't brought over).
    try
        write_membership_report(o.out_dir, ch, gt_classes, {trials.name}, ref_name, metrics, all_stims, T_mem);
    catch ME
        warning('acm:reportFailed', 'report.html: %s', ME.message);
    end

    out = struct('membership', T_mem, 'class_match', match_tbl, 'response_metrics', metrics, ...
                 'spike_metrics', spike_metrics, 'mu_trace', mu_trace_all, 'gt_classes', gt_classes, ...
                 'arm_class_map', arm_class_for, 'out_dir', o.out_dir);
    fprintf('[acm] done -> %s\n', o.out_dir);
end

% ======================================================================
%  REPORT.HTML
% ======================================================================
function write_membership_report(out_dir, ch, gt_classes, trial_names, ref_name, metrics, all_stims, T_mem)
    report_path = fullfile(out_dir, 'report.html');
    fid = fopen(report_path, 'w');
    c = onCleanup(@() fclose(fid));
    fprintf(fid, ['<!doctype html><meta charset="utf-8"><title>membership analysis ch%d</title>' ...
        '<style>body{font:13px/1.4 -apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#111}' ...
        'h2{border-bottom:2px solid #333;margin-top:40px}h3{margin:18px 0 6px}' ...
        'table{border-collapse:collapse;margin:8px 0}td,th{border:1px solid #bbb;padding:3px 7px;text-align:right}' ...
        'th{background:#eee}td.l,th.l{text-align:left}img{max-width:100%%;border:1px solid #ddd;margin:4px 0}' ...
        '.grid td{padding:2px 6px;font-size:11px}.small{color:#666;font-size:11px}</style>'], ch);
    fprintf(fid, '<h1>Membership analysis ch%d</h1><p class=small>%s<br>trials: %s (ref: %s)</p>', ...
        ch, esc_local(out_dir), esc_local(strjoin(trial_names, ', ')), esc_local(ref_name));
    fprintf(fid, '<p>Sections: %s</p>', strjoin(arrayfun(@(g) sprintf('<a href="#gt%d">gt%d</a>',g,g), ...
        gt_classes, 'UniformOutput', false), ' &middot; '));

    other_trials = setdiff(trial_names, ref_name, 'stable');
    for gi = 1:numel(gt_classes)
        gtc = gt_classes(gi);
        fprintf(fid, '<h2 id="gt%d">ch%d &nbsp; ground-truth class %d</h2>', gtc, ch, gtc);

        html_table_local(fid, T_mem(T_mem.gt_class==gtc,:), 'membership (per trial)');

        cm_csv = fullfile(out_dir, 'membership', sprintf('class_match_ch%d.csv', ch));
        if exist(cm_csv, 'file')
            Tc = readtable(cm_csv);
            if ismember('gt_class', Tc.Properties.VariableNames)
                html_table_local(fid, Tc(Tc.gt_class==gtc,:), 'class matching (headline)');
            end
        end
        rs_csv = fullfile(out_dir, 'response', sprintf('responsive_summary_ch%d_gt%d.csv', ch, gtc));
        if exist(rs_csv, 'file')
            html_table_local(fid, readtable(rs_csv), 'responsive-stimulus outcome (n_gained/n_lost/jaccard vs ref)');
        end

        fprintf(fid, '<h3>figures</h3>');
        imgs = { ...
            sprintf('membership/waveform_overlay_ch%d_gt%d.png', ch, gtc), ...
            sprintf('response/responsive_count_ch%d_gt%d.png', ch, gtc), ...
            sprintf('response/trends_ch%d_gt%d.png', ch, gtc), ...
            sprintf('response/change_summary_ch%d_gt%d.png', ch, gtc), ...
            fullfile('membership','spike_metrics', sprintf('gt%d',gtc), sprintf('spike_metrics_ch%d_gt%d_ch%d.png', ch, gtc, ch))};
        for ii = 1:numel(other_trials)
            imgs{end+1} = sprintf('membership/class_match_ch%d_%s.png', ch, ...
                matlab.lang.makeValidName(other_trials{ii}));
        end
        for img = imgs
            fp = fullfile(out_dir, img{1});
            if exist(fp, 'file'), fprintf(fid, '<img src="%s">', strrep(img{1}, filesep, '/')); end
        end

        % ---- stimulus grid: zscore per trial + responsive dot, YOUR stim order ----
        if ~isempty(all_stims)
            fprintf(fid, ['<h3>stimulus grid <span class=small>(stim_list order; ' ...
                'cell = zscore, &bull; = responsive)</span></h3><table class=grid><tr><th class=l>stim</th>']);
            for k = 1:numel(trial_names), fprintf(fid, '<th>%s</th>', esc_local(trial_names{k})); end
            fprintf(fid, '</tr>');
            for si = 1:numel(all_stims)
                snum = all_stims(si);
                fprintf(fid, '<tr><td class=l>%d</td>', snum);
                for k = 1:numel(trial_names)
                    val = metric_lookup(metrics, gtc, trial_names{k}, snum, 'zscore');
                    r = metrics([metrics.gt_class]==gtc & strcmp({metrics.trial},trial_names{k}) & [metrics.stim]==snum);
                    dot = '';
                    if ~isempty(r) && r(1).m.min_spk_test && r(1).m.p_test && r(1).m.good_lat && abs(val) >= 3
                        dot = ' &bull;';
                    end
                    fprintf(fid, '<td>%s%s</td>', num2str_local(val), dot);
                end
                fprintf(fid, '</tr>');
            end
            fprintf(fid, '</table>');
        end
    end
    fprintf(fid, '<hr><p class=small>generated %s</p>', char(datetime('now')));
    fprintf('[acm] report -> %s\n', report_path);
end

function html_table_local(fid, T, caption)
    if isempty(T), return; end
    fprintf(fid, '<h3>%s</h3><table><tr>', esc_local(caption));
    vn = T.Properties.VariableNames;
    for k = 1:numel(vn), fprintf(fid, '<th class=l>%s</th>', esc_local(vn{k})); end
    fprintf(fid, '</tr>');
    for r = 1:height(T)
        fprintf(fid, '<tr>');
        for k = 1:numel(vn)
            v = T.(vn{k})(r);
            if iscell(v), v = v{1}; end
            fprintf(fid, '<td class=l>%s</td>', esc_local(num2str_local(v)));
        end
        fprintf(fid, '</tr>');
    end
    fprintf(fid, '</table>');
end

function s = num2str_local(v)
    if isstring(v) || ischar(v), s = char(v);
    elseif islogical(v), s = mat2str(v);
    elseif isnumeric(v)
        if isnan(v), s = ''; elseif v==round(v), s = sprintf('%d',v); else, s = sprintf('%.3g',v); end
    else, s = char(string(v));
    end
end

function s = esc_local(s)
    s = char(string(s));
    s = strrep(strrep(strrep(s,'&','&amp;'),'<','&lt;'),'>','&gt;');
end

% ======================================================================
%  RECONSTRUCTION (post-onset only, identical for plain/blanks by design)
% ======================================================================
function omap = build_onset_map(root)
    omap = containers.Map('KeyType', 'double', 'ValueType', 'any');
    S = load(fullfile(root, 'stimulus.mat'));       stimulus = S.stimulus;
    F = load(fullfile(root, 'finalevents.mat'));    pics_onset = F.pics_onset;
    E = load(fullfile(root, 'experiment_properties_online3.mat'), 'scr_end_cell', 'scr_config_cell');
    Nscr = min(numel(stimulus), numel(E.scr_end_cell));
    have_cfg = isfield(E, 'scr_config_cell');
    for scri = 1:Nscr
        po = pics_onset{scri}; st = stimulus{scri};
        p2u = [];
        if have_cfg && numel(E.scr_config_cell) >= scri && isfield(E.scr_config_cell{scri}, 'pics2use')
            p2u = E.scr_config_cell{scri}.pics2use;
        end
        for j = 1:numel(st)
            snum = resolve_stim_id(st(j).ID, p2u); terna = st(j).terna_onset;
            if ndims(po) == 3
                ts = po(sub2ind(size(po), terna(:,1), terna(:,2), terna(:,3)));
            else
                M = po(terna(:,1), terna(:,3)); ts = M(1:length(M));
            end
            ts = ts(:).';
            if ~isKey(omap, snum), omap(snum) = []; end
            omap(snum) = [omap(snum), ts];
        end
    end
end

function snum = resolve_stim_id(raw_id, pics2use)
% Sessions recorded before 2024-11-18 ran an older acquisition-code
% create_stimulus_online.m (codes_emu repo) that stored st(j).ID as
% ipic + iISI/10 -- ipic being the picture's LOCAL loop position within
% that screening block's own picture list (scr_config.pics2use), plus a
% decimal tag for the ISI condition -- instead of the real catalog picture
% id. Commit fec5283 fixed this to store pics2use(ipic) directly (a plain
% integer, matching the raster's own positional stimulus indexing). This
% is what produces the ".1"-suffixed ids seen on some archived sessions
% (e.g. pt 12 emu 18 ch 300, subtask CategLocaliz, stimulus.mat dated
% 2024-03) -- confirmed against the actual pre-fix source line, not
% inferred.
%
% Detect the old encoding by a non-zero fractional part. When the
% session's own scr_config_cell{.}.pics2use is available (it always has
% been, in both old- and new-format sessions checked so far), back out
% the true id exactly the way the fixed code computes it --
% pics2use(ipic) -- rather than just rounding: round(raw_id) is only
% correct when pics2use happens to equal 1:N for that session, which is
% NOT guaranteed (only verified true for pt 12 emu 18 ch 300 itself, by
% cross-checking its own scr_config_cell.pics2use).
    if raw_id == round(raw_id)
        snum = raw_id;              % new-format session: already the real catalog id
        return;
    end
    ipic = floor(raw_id);           % old-format: integer part is the local loop position
    if ~isempty(pics2use) && ipic >= 1 && ipic <= numel(pics2use)
        snum = pics2use(ipic);
    else
        warning('acm:oldFormatStimIdNoPics2use', ['st(j).ID=%.4g looks like the pre-2024-11-18 ' ...
            'ipic+iISI/10 encoding but scr_config_cell.pics2use is unavailable for this screening ' ...
            'block -- falling back to round(), which is only exact when pics2use==1:N for this ' ...
            'session (unverified here).'], raw_id);
        snum = round(raw_id);
    end
end

function [match_ok, n_times, n_details] = check_class_count_agreement(times_file, grapes_file, ch, class_num)
% Does a trial's times file agree with its OWN grapes raster about how many
% spikes this class has? n_times: raw whole-session count from
% times_file's cluster_class. n_details: rasters.chan<ch>.details.class<N>
% in the grapes file (the same number plot_channel_grapes prints as
% "(N spks)" on a raster title -- confirmed by direct comparison to be
% populated from cluster_class at raster-build time, so agreement here
% means "the times file hasn't changed since this grapes file was built."
% match_ok=false (with n_times/n_details left NaN where unavailable) is the
% safe default -- no times file, no cluster_class field, or no matching
% details field all count as "don't trust it," never silently full-scope.
    match_ok = false; n_times = NaN; n_details = NaN;
    if isempty(times_file) || ~exist(times_file, 'file'), return; end
    T = load(times_file, 'cluster_class');
    if ~isfield(T, 'cluster_class'), return; end
    n_times = nnz(T.cluster_class(:,1) == class_num);
    if isempty(grapes_file) || ~exist(grapes_file, 'file'), return; end
    G = load(grapes_file, 'rasters');
    cf = sprintf('chan%d', ch); fld = sprintf('class%d', class_num);
    if ~isfield(G,'rasters') || ~isfield(G.rasters,cf) || ~isfield(G.rasters.(cf),'details') ...
            || ~isfield(G.rasters.(cf).details, fld)
        return;
    end
    n_details = G.rasters.(cf).details.(fld);
    match_ok = (n_times == n_details);
end

function [times_file, grapes_file, cc] = trial_files_for_name(tname, ref, ref_name, trials)
% Map a trial NAME back to its times_file/grapes_file/windowed cc -- used
% by the spike_metrics augmentation, which only has collect_sorting_
% metrics' output table (trial names as strings) to work from, not the
% original trials struct index.
    times_file = ''; grapes_file = ''; cc = zeros(0,2);
    if strcmp(tname, ref_name)
        times_file = ref.times_file; grapes_file = ref.grapes_file; cc = ref.cc;
        return;
    end
    k = find(strcmp({trials.name}, tname), 1);
    if ~isempty(k)
        times_file = trials(k).times_file; grapes_file = trials(k).grapes_file; cc = trials(k).cc;
    end
end

function cc = full_cc_from_times(times_file)
% Every non-zero class's spike times straight from the times file's OWN
% cluster_class -- full, whole-session, [class time_ms] shape, same as
% reconstruct_class_cc's windowed output but unwindowed. For match_cc,
% which needs every class at once (it reports where a GT class's spikes
% land across ALL of the arm's classes, not just the matched one).
    cc = zeros(0,2);
    if isempty(times_file) || ~exist(times_file, 'file'), return; end
    T = load(times_file, 'cluster_class');
    if ~isfield(T, 'cluster_class'), return; end
    keep = T.cluster_class(:,1) ~= 0;
    cc = [T.cluster_class(keep,1), T.cluster_class(keep,2)];
end

function t = full_class_times_from_times_file(times_file, class_num)
% This class's spike times straight from the times file's OWN cluster_class
% -- the full, whole-session population, same shape/units (ms) as
% reconstruct_class_cc's windowed output, so every caller that already
% consumes a [class time_ms]-derived time vector (membership_sets,
% nearest_match, ...) works on this unchanged. Only meaningful to use when
% check_class_count_agreement said this times file is trustworthy.
    t = zeros(0,1);
    if isempty(times_file) || ~exist(times_file, 'file'), return; end
    T = load(times_file, 'cluster_class');
    if ~isfield(T, 'cluster_class'), return; end
    t = T.cluster_class(T.cluster_class(:,1)==class_num, 2);
end

function cc = reconstruct_class_cc(grapes_file, ch, omap)
    cc = zeros(0,2);
    G = load(grapes_file); cf = ['chan' num2str(ch)];
    if ~isfield(G,'rasters') || ~isfield(G.rasters,cf), return; end
    R = G.rasters.(cf); fn = fieldnames(R);
    cls = sort(cellfun(@(x) sscanf(x,'class%d'), fn(~cellfun('isempty', regexp(fn,'^class\d+$','once')))))';
    allt=[]; allc=[];
    for ci=1:numel(cls)
        C = sprintf('class%d', cls(ci)); stimc = R.(C).stim;
        for snum=1:numel(stimc)
            tr = stimc{snum}; if isempty(tr), continue; end
            if ~iscell(tr), tr = num2cell(tr,2); end
            if ~isKey(omap,snum), continue; end
            O = omap(snum); if numel(O)~=numel(tr), continue; end
            for t=1:numel(tr)
                rel = tr{t}(:).'; rel = rel(rel>=0 & rel<9000);
                if isempty(rel), continue; end
                allt=[allt, rel+O(t)]; allc=[allc, repmat(cls(ci),1,numel(rel))];
            end
        end
    end
    if isempty(allt), return; end
    [allt,ord]=sort(allt); allc=allc(ord);
    keep=[true, ~(diff(allt)<=0.05 & diff(allc)==0)];
    cc=[allc(keep)', allt(keep)'];
end

function t = reconstruct_mu_times(grapes_file, ch, omap)
    t = [];
    G = load(grapes_file); cf = ['chan' num2str(ch)];
    if ~isfield(G,'rasters') || ~isfield(G.rasters,cf) || ~isfield(G.rasters.(cf),'mu'), return; end
    stimc = G.rasters.(cf).mu.stim; allt=[];
    for snum=1:numel(stimc)
        tr = stimc{snum}; if isempty(tr), continue; end
        if ~iscell(tr), tr = num2cell(tr,2); end
        if ~isKey(omap,snum), continue; end
        O = omap(snum); if numel(O)~=numel(tr), continue; end
        for k=1:numel(tr)
            rel = tr{k}(:).'; rel = rel(rel>=0 & rel<9000);
            if ~isempty(rel), allt=[allt, rel+O(k)]; end
        end
    end
    allt = sort(allt);
    if ~isempty(allt), allt = allt([true, diff(allt)>0.05]); end
    t = allt(:);
end

function rows = reconstruct_class_detail(grapes_file, ch, class_id, omap)
% Like reconstruct_class_cc, but for ONE class, keeping (snum, trial_idx, rel)
% alongside the absolute time -- so any abs-time subset (matched/gt_only/
% arm_only, already computed elsewhere) can be re-binned straight back into
% stimulus-raster rows without recomputing anything. Same post-onset-only,
% same dedup rule, so `[rows.abs]` is bit-identical to the corresponding
% column of reconstruct_class_cc's output for this class -- exact membership
% via `ismember(...,'abs')` is safe, no tolerance needed.
    rows = struct('snum',{}, 'trial_idx',{}, 'rel',{}, 'abs',{});
    G = load(grapes_file); cf = ['chan' num2str(ch)];
    if ~isfield(G,'rasters') || ~isfield(G.rasters,cf), return; end
    R = G.rasters.(cf); C = sprintf('class%d', class_id);
    if ~isfield(R, C), return; end
    stimc = R.(C).stim;
    a_snum=[]; a_tidx=[]; a_rel=[]; a_abs=[];
    for snum = 1:numel(stimc)
        tr = stimc{snum}; if isempty(tr), continue; end
        if ~iscell(tr), tr = num2cell(tr,2); end
        if ~isKey(omap, snum), continue; end
        O = omap(snum); if numel(O) ~= numel(tr), continue; end
        for t = 1:numel(tr)
            rel = tr{t}(:).'; rel = rel(rel>=0 & rel<9000);
            if isempty(rel), continue; end
            a_snum = [a_snum, repmat(snum,1,numel(rel))];
            a_tidx = [a_tidx, repmat(t,1,numel(rel))];
            a_rel  = [a_rel, rel];
            a_abs  = [a_abs, rel + O(t)];
        end
    end
    if isempty(a_abs), return; end
    [a_abs, ord] = sort(a_abs);
    a_snum = a_snum(ord); a_tidx = a_tidx(ord); a_rel = a_rel(ord);
    keep = [true, diff(a_abs) > 0.05];
    rows = struct('snum', num2cell(a_snum(keep)), 'trial_idx', num2cell(a_tidx(keep)), ...
                  'rel', num2cell(a_rel(keep)), 'abs', num2cell(a_abs(keep)));
end

function run_plot_grapes_as_online(grapes_file, is_rescue, ch, stim_order, dest_dir)
% Call the REAL plot_grapes_as_online on an existing grapes file (never
% synthesized for arm/gt -- their files already exist), then move its
% output folder into dest_dir so every raster set lands in the same
% trial_ras_dir/<set>/ layout. is_rescue picks grapes_blanks vs
% grapes_blanks_rescue (plot_grapes_as_online derives the filename itself
% from use_blanks/rescue, never from a path -- so grapes_file's own name is
% only used here to find target_dir and to tell rescue apart).
    target_dir = fileparts(grapes_file);
    if isempty(target_dir), target_dir = pwd; end
    try
        plot_grapes_as_online('target_dir', target_dir, 'channels2plot', ch, ...
            'use_blanks', true, 'rescue', is_rescue, 'stim_list', stim_order, ...
            'order_by_rank', false, 'copy2miniscrfolder', false, 'grapes_offline', true);
    catch ME
        warning('%s', ['run_plot_grapes_as_online failed for ' grapes_file ': ' ME.message]);
        return
    end
    suffix = 'offline_grapes_new_order_results';
    if is_rescue, suffix = [suffix '_rescue_post']; end
    src = fullfile(target_dir, suffix);
    if ~exist(src, 'dir')
        d = dir(fullfile(target_dir, 'offline_grapes_new_order_results*'));
        d = d([d.isdir]);
        if isempty(d), return; end
        [~, k] = max([d.datenum]); src = fullfile(target_dir, d(k).name);
    end
    if ~exist(dest_dir, 'dir'), mkdir(dest_dir); end
    movefile(fullfile(src, '*'), dest_dir);
    rmdir(src);
end

function nm = nm_for_ch(root, ch)
    NSx = getfield(load(fullfile(root, 'NSx.mat')), 'NSx');
    pos = find(arrayfun(@(x) x.chan_ID == ch, NSx), 1);
    if isempty(pos), nm = sprintf('ch%d', ch); else, nm = NSx(pos).output_name; end
end

function gf = synth_grapes_for_subset(abs_times, ch, root, template_grapes_file, out_dir, nm, circshiftblanks)
% Build a grapes_blanks.mat for a derived spike-time set that exists nowhere
% in any real grapes file (matched/gt_only/arm_only, or a within_channel-
% check flagged/rescued subset) by calling the REAL update_grapes_blanks --
% with domu=true, its own intended "just build mu" mode, not a workaround.
% A derived subset is a single undifferentiated spike-time list, not a
% clustering result -- there is no "class" to assign it to, so this no
% longer manufactures a class1 and aliases mu from it (that hack was the
% bug: domu=false never fills .details.mu, which plot_channel_grapes.m
% reads unconditionally for every unit -> "Unrecognized field name 'mu'"
% the instant it processed the aliased mu. Verified fixed by calling the
% real domu=true path instead, which fills .mu/.details.mu correctly on
% its own -- no class1, no manual field copying, no hack.)
%
% Returns gf = '' (no file written, nothing plotted) when abs_times is
% empty -- a genuinely empty overlap set is a real, valid result, not an
% error; callers must check for '' and skip plotting rather than crash.
    gf = '';
    abs_times = sort(abs_times(:)');
    if isempty(abs_times), return; end
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end
    T = load(template_grapes_file, 'ImageNames', 'exp_type');
    S = load(fullfile(root, 'stimulus.mat'));    stimulus = S.stimulus;
    F = load(fullfile(root, 'finalevents.mat')); pics_onset = F.pics_onset;
    E = load(fullfile(root, 'experiment_properties_online3.mat'), 'scr_config_cell', 'scr_end_cell');
    Nscr = min(numel(stimulus), numel(E.scr_end_cell));

    grapes = struct('time_pre', 1000, 'time_pos', 2000, 'exp_type', T.exp_type, 'ImageNames', {T.ImageNames});
    for scri = 1:Nscr
        grapes = update_grapes_blanks(grapes, pics_onset{scri}, F.seq_beg_blanks_cell{scri}, ...
            stimulus{scri}, {abs_times}, ch, {nm}, 1, {[]}, ...
            E.scr_config_cell{scri}.pics2use, scri, circshiftblanks, false);
    end
    gf = fullfile(out_dir, 'grapes_blanks.mat');
    save(gf, '-struct', 'grapes');
end

% ======================================================================
%  1. MEMBERSHIP
% ======================================================================
function [arm_class, arm_t] = match_arm_class(ref_cc, ref_class, arm_cc)
    arm_class = NaN; arm_t = [];
    ref_t = ref_cc(ref_cc(:,1)==ref_class, 2);
    if isempty(ref_t) || isempty(arm_cc), return; end
    arm_classes = unique(arm_cc(:,1))';
    best_recall = 0;
    for k = arm_classes
        at = arm_cc(arm_cc(:,1)==k, 2);
        gm = nearest_match(ref_t, at, 0.5);
        recall = nnz(gm>0) / numel(ref_t);
        if recall > best_recall, best_recall = recall; arm_class = k; arm_t = at; end
    end
end

function [matched_gt, matched_arm, gt_only, arm_only] = membership_sets(gt_t, arm_t, tol)
    gm = nearest_match(gt_t, arm_t, tol);      % greedy nearest, NOT yet one-to-one unique
    [gm_u, gt_order] = enforce_one_to_one(gt_t, arm_t, gm, tol);
    matched_gt  = gt_t(gt_order(gm_u>0));
    matched_arm = arm_t(gm_u(gm_u>0));
    gt_only  = gt_t(gt_order(gm_u==0));
    used = false(numel(arm_t),1); used(gm_u(gm_u>0)) = true;
    arm_only = arm_t(~used);
end

function [gm_u, order] = enforce_one_to_one(gt_t, arm_t, gm, tol)
% Greedy one-to-one: process GT spikes in order of match distance (tightest
% first) so no arm spike is claimed by two GT spikes.
    order = (1:numel(gt_t))';
    dist = nan(size(gm));
    ok = gm>0; dist(ok) = abs(gt_t(ok) - arm_t(gm(ok)));
    [~, seq] = sort(dist);   % NaNs (unmatched) sort last
    gm_u = zeros(size(gm));
    claimed = false(numel(arm_t),1);
    for ii = seq(:)'
        if gm(ii)==0, continue; end
        if ~claimed(gm(ii))
            gm_u(ii) = gm(ii); claimed(gm(ii)) = true;
        end
    end
end

function plot_membership_set(gtc, trial_name, arm_class, gt_t, arm_t, matched_gt, gt_only, arm_only, ...
        out_dir, ch, scope, n_gt_full, n_arm_full)
% gt_t/arm_t already reflect the CHOSEN scope (n_gt/n_arm below == whatever
% is actually drawn) -- when scope=='windowed' (times-vs-grapes mismatch on
% either side), the venn is still sized/drawn off these (windowed) numbers
% only; n_gt_full/n_arm_full are shown as a text annotation alongside, never
% used for sizing, so the full-session numbers stay visible without making
% the comparison unfair.
    n_gt = numel(gt_t); n_arm = numel(arm_t); n_m = numel(matched_gt);
    tag = sprintf('ch%d_gt%d_%s', ch, gtc, trial_name);
    full_note = '';
    if strcmp(scope, 'windowed')
        full_note = sprintf('  [times/grapes mismatch -> windowed shown; full session: GT=%s, %s=%s]', ...
            num2str_local2(n_gt_full), trial_name, num2str_local2(n_arm_full));
    end

    % raster: 3 rows of tick marks (matched / gt-only / arm-only) vs spike TIME
    f = figure('Visible','off','Position',[100 100 1100 380],'Color','w');
    ax = axes(f); hold(ax,'on');
    plot(ax, matched_gt/1000/60, ones(size(matched_gt)), '|', 'Color',[0.2 0.6 0.2]);
    plot(ax, gt_only/1000/60, 2*ones(size(gt_only)), '|', 'Color',[0.8 0.2 0.2]);
    plot(ax, arm_only/1000/60, 3*ones(size(arm_only)), '|', 'Color',[0.2 0.3 0.8]);
    ylim(ax,[0.5 3.5]); yticks(ax,1:3);
    yticklabels(ax, {sprintf('matched (%d)',n_m), sprintf('GT-only (%d)',numel(gt_only)), ...
                     sprintf('%s-only (%d)',trial_name,numel(arm_only))});
    set(ax,'TickLabelInterpreter','none'); xlabel(ax,'time (min)'); grid(ax,'on');
    title(ax, sprintf('ch%d gt-class%d vs %s (class%d)  |  spikes over the session  (%s)%s', ...
          ch,gtc,trial_name,arm_class,scope,full_note), 'Interpreter','none','FontSize',9);
    save_fig_local(f, fullfile(out_dir, ['raster_' tag '.png']));

    % venn (2-circle, proportional-ish -- matplotlib-free manual draw)
    f = figure('Visible','off','Position',[100 100 520 460],'Color','w');
    ax = axes(f); hold(ax,'on'); axis(ax,'equal'); axis(ax,'off');
    r1 = sqrt(n_gt); r2 = sqrt(n_arm); rr = max([r1 r2 1]);
    r1n = 0.9*r1/rr; r2n = 0.9*r2/rr;
    ov = n_m/max(min(n_gt,n_arm),1);           % 0..1 overlap fraction
    d = (r1n+r2n)*(1-0.85*ov);                 % center separation, shrinks as overlap grows
    th = linspace(0,2*pi,200);
    fill(ax, -d/2+r1n*cos(th), r1n*sin(th), [0.85 0.3 0.3], 'FaceAlpha',0.45,'EdgeColor',[0.6 0.1 0.1]);
    fill(ax,  d/2+r2n*cos(th), r2n*sin(th), [0.3 0.4 0.85], 'FaceAlpha',0.45,'EdgeColor',[0.1 0.1 0.6]);
    text(ax, -d/2-r1n*0.6, 0, sprintf('GT-only\n%d',numel(gt_only)), 'HorizontalAlignment','center','FontSize',10);
    text(ax,  d/2+r2n*0.6, 0, sprintf('%s-only\n%d',trial_name,numel(arm_only)), 'HorizontalAlignment','center','FontSize',10,'Interpreter','none');
    text(ax, 0, 0, sprintf('matched\n%d',n_m), 'HorizontalAlignment','center','FontSize',11,'FontWeight','bold');
    title(ax, sprintf('ch%d gt-class%d vs %s(class%d)  (%s)\nGT n=%d, arm n=%d, %.0f%% of GT recovered%s', ...
          ch,gtc,trial_name,arm_class,scope,n_gt,n_arm,100*n_m/max(n_gt,1),full_note), ...
          'Interpreter','none','FontSize',9);
    save_fig_local(f, fullfile(out_dir, ['venn_' tag '.png']));
end

function s = num2str_local2(v)
    if isnan(v), s = 'n/a'; else, s = sprintf('%d', round(v)); end
end

% ======================================================================
%  2. RESPONSE METRICS (real, via characterize_grapes_responses)
% ======================================================================
function T = run_characterize_grapes(grapes_file, ch, stim_list, root)
% Call the REAL characterize_grapes_responses (-> create_responses_data_
% parallel, same one plot_grapes_as_online/do_structure_* use) on ONE
% grapes file, restricted to this channel + stim_list. cd's into the
% file's own folder (characterize_grapes_responses globs grapes*.mat from
% pwd) and restores cwd after. Returns its metrics table, or an empty
% table if the file/tag wasn't found.
    D = fileparts(grapes_file);
    if isempty(D), D = pwd; end
    old = pwd; c = onCleanup(@() cd(old));
    cd(D);
    [~, gbase, gext] = fileparts(grapes_file);
    res = characterize_grapes_responses('channels2plot', ch, 'stim_list', stim_list, ...
        'session_root', root, 'grapes_files', {[gbase gext]}, ...
        'outfile', fullfile(D, 'grapes_response_metrics.mat'));
    gtag = matlab.lang.makeValidName(gbase);
    if isfield(res, gtag) && isfield(res.(gtag), 'metrics')
        T = res.(gtag).metrics;
    else
        T = table();
    end
end

function row = summarize_wc_response(gf, ch, stim_list, root, check, subset, trial, n_spikes, zthr)
% Real response metrics (characterize_grapes_responses, on the SAME synth
% file already built for the raster) for one within_channel-check
% flagged/rescued population, reduced to the same responsive-gate summary
% validate_recluster_vs_handselect uses (min_spk_test & p_test & good_lat
% & |zscore|>=zthr): does this discarded/rescued population actually carry
% a stimulus response, and how much.
    row = struct('check', check, 'subset', subset, 'trial', trial, 'n_spikes', n_spikes, ...
        'n_responsive', 0, 'frac_responsive', NaN, 'mean_zscore', NaN, 'max_zscore', NaN, ...
        'mean_resp_delta_fr_hz', NaN);
    T = run_characterize_grapes(gf, ch, stim_list, root);
    if isempty(T), return; end
    sel = strcmp(T.class, 'mu');   % synth files are single-population -> always 'mu'
    T = T(sel, :);
    if isempty(T), return; end
    gate = T.min_spk_test & T.p_test & T.good_lat & (abs(T.zscore) >= zthr);
    row.n_responsive = nnz(gate);
    row.frac_responsive = mean(gate);
    row.mean_zscore = mean(T.zscore, 'omitnan');
    row.max_zscore = max(T.zscore, [], 'omitnan');
    row.mean_resp_delta_fr_hz = mean(T.resp_delta_fr_hz, 'omitnan');
end

function pairwise_metric_plots(metrics, trial_names, ref_name, gt_classes, out_dir, ch)
% Every pair (incl. arm-vs-arm, kept) -- but every panel ALSO marks where
% ground truth sits: a per-stimulus crosshair at (gt_val, gt_val), which
% always falls ON the y=x parity line since it's the same metric on both
% axes. Distance from a real point to its own crosshair shows how far that
% (trialA,trialB) pair drifted from GT for that stimulus -- not just from
% each other.
    if isempty(metrics), return; end
    flds = {'zscore','mu_base','std_base','median_post','p_value_sign','p_ranksum','auroc', ...
           'resp_delta_fr_hz','mean_fr_base_hz','mean_fr_post_hz','spk_post_mean','spk_post_std', ...
           'frac_active_trials','peak_ifr','peak_ifr_lat','onset_ms','duration_ms','tons_ms', ...
           'ntrials','ifr_thr','first_spk_lat_med_ms'};
    for gi = 1:numel(gt_classes)
        gtc = gt_classes(gi);
        sel = [metrics.gt_class] == gtc;
        Mgt = metrics(sel);
        present = unique({Mgt.trial});
        present = trial_names(ismember(trial_names, present));
        Ref = Mgt(strcmp({Mgt.trial}, ref_name));
        [stimRef, iRef] = sort([Ref.stim]);
        for a = 1:numel(present)
            for b = a+1:numel(present)
                tA = present{a}; tB = present{b};
                A = Mgt(strcmp({Mgt.trial}, tA)); B = Mgt(strcmp({Mgt.trial}, tB));
                [stimA, iA] = sort([A.stim]); [stimB, iB] = sort([B.stim]);
                common = intersect(stimA, stimB, 'stable');
                if isempty(common), continue; end
                f = figure('Visible','off','Position',[100 100 1700 1400],'Color','w');
                tl = tiledlayout(f, 'flow', 'Padding','loose','TileSpacing','loose');
                title(tl, sprintf('ch%d gt-class%d: %s vs %s  (n=%d common stim, + = %s)', ...
                     ch,gtc,tA,tB,numel(common),ref_name), 'Interpreter','none');
                for fi = 1:numel(flds)
                    ax = nexttile(tl); hold(ax,'on');
                    xv = arrayfun(@(s) A(iA(stimA==s)).m.(flds{fi}), common);
                    yv = arrayfun(@(s) B(iB(stimB==s)).m.(flds{fi}), common);
                    good = isfinite(xv) & isfinite(yv);
                    scatter(ax, xv(good), yv(good), 22, 'filled', 'MarkerFaceAlpha',0.6, 'HandleVisibility','off');
                    if any(good)
                        lo = min([xv(good) yv(good)]); hi = max([xv(good) yv(good)]);
                        if isfinite(lo) && isfinite(hi) && hi>lo
                            plot(ax,[lo hi],[lo hi],'k--', 'HandleVisibility','off');
                        end
                    end
                    if ~isempty(Ref)
                        gv = arrayfun(@(s) metric_lookup_row(Ref, iRef, stimRef, s, flds{fi}), common);
                        gg = isfinite(gv);
                        if any(gg)
                            plot(ax, gv(gg), gv(gg), '+', 'Color',[0.85 0.1 0.1], 'MarkerSize',7, ...
                                'LineWidth',1.3, 'DisplayName', ref_name);
                        end
                    end
                    grid(ax,'on'); axis(ax,'square');
                    xlabel(ax,tA,'Interpreter','none'); ylabel(ax,tB,'Interpreter','none');
                    title(ax, flds{fi}, 'Interpreter','none','FontSize',9);
                    if fi==1, legend(ax,'Location','best','FontSize',7); end
                end
                save_fig_local(f, fullfile(out_dir, sprintf('pairwise_ch%d_gt%d_%s_vs_%s.png', ch,gtc,tA,tB)));
            end
        end
    end
    % also dump the raw metrics table
    rows = struct2table(metrics, 'AsArray', true);
    if ismember('m', rows.Properties.VariableNames)
        mm = struct2table([metrics.m]);
        rows = [removevars(rows,'m'), mm];
    end
    writetable(rows, fullfile(out_dir, sprintf('response_metrics_ch%d.csv', ch)));
end

function change_vs_ref_summary(metrics, T_mem, trial_names, ref_name, gt_classes, out_dir, ch)
% For ONE ground-truth class at a time (never mixed across classes), every
% trial's response change vs the reference: the per-stim deltas as actual
% numbers (not just a scatter to eyeball) AND the average change across
% stimuli, alongside the same-class spike-count match/%change from the
% membership step -- one place with everything asked for: match count,
% %increase/decrease in spikes, average stim change, per-stim change.
    if isempty(metrics), return; end
    flds = {'zscore','mu_base','std_base','median_post','p_value_sign','p_ranksum','auroc', ...
           'resp_delta_fr_hz','mean_fr_base_hz','mean_fr_post_hz','spk_post_mean','spk_post_std', ...
           'frac_active_trials','peak_ifr','peak_ifr_lat','onset_ms','duration_ms','tons_ms', ...
           'ntrials','ifr_thr','first_spk_lat_med_ms'};
    other_trials = setdiff(trial_names, ref_name, 'stable');

    for gi = 1:numel(gt_classes)
        gtc = gt_classes(gi);
        ref_rows = metrics([metrics.gt_class]==gtc & strcmp({metrics.trial}, ref_name));
        if isempty(ref_rows), continue; end
        ref_stim = [ref_rows.stim];

        long_rows = struct('trial',{},'stim',{},'metric',{},'ref_value',{},'arm_value',{},'delta',{},'pct_change',{});
        summ_list = {};   % each entry a scalar struct with identical fields (built below) -- keeps
                          % struct2table([summ_list{:}]) safe regardless of which flds are present

        for ti = 1:numel(other_trials)
            tname = other_trials{ti};
            arm_rows = metrics([metrics.gt_class]==gtc & strcmp({metrics.trial}, tname));
            if isempty(arm_rows), continue; end
            arm_stim = [arm_rows.stim];
            common = intersect(ref_stim, arm_stim, 'stable');

            mem_row = T_mem(T_mem.gt_class==gtc & strcmp(T_mem.trial, tname), :);
            srow = struct('trial', tname, ...
                'n_gt', pick(mem_row,'n_gt'), 'n_arm', pick(mem_row,'n_arm'), ...
                'n_matched', pick(mem_row,'n_matched'), 'pct_gt_lost', pick(mem_row,'pct_gt_lost'), ...
                'pct_arm_new', pick(mem_row,'pct_arm_new'), 'n_common_stim', numel(common));

            for fi = 1:numel(flds)
                fname = flds{fi};
                d = nan(numel(common),1); pc = nan(numel(common),1);
                rv = nan(numel(common),1); av = nan(numel(common),1);
                for si = 1:numel(common)
                    s = common(si);
                    r = ref_rows([ref_rows.stim]==s).m.(fname);
                    a = arm_rows([arm_rows.stim]==s).m.(fname);
                    rv(si) = r; av(si) = a; d(si) = a - r;
                    if isfinite(r) && r~=0, pc(si) = 100*(a-r)/abs(r); end
                    long_rows(end+1) = struct('trial',tname,'stim',s,'metric',fname, ...
                        'ref_value',r,'arm_value',a,'delta',d(si),'pct_change',pc(si));
                end
                srow.(['mean_delta_' fname])   = mean(d, 'omitnan');
                srow.(['median_delta_' fname]) = median(d, 'omitnan');
                srow.(['mean_pctchange_' fname]) = mean(pc, 'omitnan');
            end
            summ_list{end+1} = srow;
        end

        if ~isempty(long_rows)
            writetable(struct2table(long_rows), fullfile(out_dir, sprintf('change_ch%d_gt%d.csv', ch, gtc)));
        end
        if ~isempty(summ_list)
            Tsumm = struct2table([summ_list{:}]);
            writetable(Tsumm, fullfile(out_dir, sprintf('change_summary_ch%d_gt%d.csv', ch, gtc)));
            plot_change_summary(Tsumm, flds, ch, gtc, ref_name, out_dir);
        end
    end
end

function v = pick(T, col)
    if isempty(T) || ~ismember(col, T.Properties.VariableNames), v = NaN; else, v = T.(col)(1); end
end

function plot_change_summary(Tsumm, flds, ch, gtc, ref_name, out_dir)
    f = figure('Visible','off','Position',[80 80 1700 1300],'Color','w');
    tl = tiledlayout(f, 'flow', 'Padding','loose','TileSpacing','loose');
    title(tl, sprintf('ch%d gt-class%d  -  change vs %s  (x = trial)', ch, gtc, ref_name), 'Interpreter','none');

    ax = nexttile(tl);
    bar(ax, categorical(Tsumm.trial), [Tsumm.n_matched, Tsumm.n_gt-Tsumm.n_matched]);
    legend(ax, {'matched','GT-only (lost)'}, 'Location','best'); grid(ax,'on');
    set(ax,'TickLabelInterpreter','none'); title(ax,'spike match count','Interpreter','none');

    ax = nexttile(tl);
    bar(ax, categorical(Tsumm.trial), [Tsumm.pct_gt_lost, Tsumm.pct_arm_new]);
    legend(ax, {'% GT lost','% arm new'}, 'Location','best'); grid(ax,'on');
    set(ax,'TickLabelInterpreter','none'); title(ax,'spike count %change','Interpreter','none');

    for fi = 1:numel(flds)
        ax = nexttile(tl);
        bar(ax, categorical(Tsumm.trial), Tsumm.(['mean_delta_' flds{fi}]));
        grid(ax,'on'); set(ax,'TickLabelInterpreter','none');
        title(ax, ['mean ' char(916) ' ' flds{fi}], 'Interpreter','none', 'FontSize', 9);
    end
    save_fig_local(f, fullfile(out_dir, sprintf('change_summary_ch%d_gt%d.png', ch, gtc)));
end

% ======================================================================
%  2a2. RESPONSIVE-STIMULUS SUMMARY (real pipeline gate, vs ref)
% ======================================================================
function compute_responsive_summary(metrics, gt_classes, trial_names, ref_name, zthr, out_dir, ch)
% Same responsive definition + n_gained/n_lost/jaccard_resp logic as
% validate_recluster_vs_handselect's build_compare/compare_plots: a
% (trial,gt_class,stim) counts as responsive when the real pipeline's own
% gate holds (min_spk_test & p_test & good_lat) AND |zscore| >= zthr. Answers
% "did rescue restore stimuli masking made non-responsive" directly, not
% just "did the zscore number move."
    if isempty(metrics), return; end
    other_trials = setdiff(trial_names, ref_name, 'stable');
    trial_order = [ref_name, other_trials];
    for gi = 1:numel(gt_classes)
        gtc = gt_classes(gi);
        sel = [metrics.gt_class] == gtc;
        Mgt = metrics(sel);
        if isempty(Mgt), continue; end
        stims = unique([Mgt.stim]);
        rows = struct('trial',{},'gt_class',{},'n_responsive',{},'frac_responsive',{}, ...
            'n_gained',{},'n_lost',{},'jaccard_resp',{},'mean_zscore',{});
        R0 = [];
        for ti = 1:numel(trial_order)
            tname = trial_order{ti};
            gate = false(size(stims));
            z = nan(size(stims));
            for si = 1:numel(stims)
                r = Mgt(strcmp({Mgt.trial}, tname) & [Mgt.stim]==stims(si));
                if isempty(r), continue; end
                m = r(1).m;
                gate(si) = m.min_spk_test && m.p_test && m.good_lat && (abs(m.zscore) >= zthr);
                z(si) = m.zscore;
            end
            if ti == 1, R0 = gate; end
            inter = nnz(R0 & gate); uni = nnz(R0 | gate);
            rows(end+1) = struct('trial', tname, 'gt_class', gtc, 'n_responsive', nnz(gate), ...
                'frac_responsive', mean(gate), 'n_gained', nnz(gate & ~R0), ...
                'n_lost', nnz(R0 & ~gate), 'jaccard_resp', inter/max(uni,1), ...
                'mean_zscore', mean(z,'omitnan'));
        end
        Tr = struct2table(rows);
        writetable(Tr, fullfile(out_dir, sprintf('responsive_summary_ch%d_gt%d.csv', ch, gtc)));

        f = figure('Visible','off','Position',[100 100 max(900,240*numel(trial_order)) 460],'Color','w');
        ax = axes(f);
        bar(ax, categorical(Tr.trial, Tr.trial), Tr.n_responsive);
        grid(ax,'on'); set(ax,'TickLabelInterpreter','none');
        ylabel(ax, sprintf('# responsive stim (|z|>=%.1f & pipeline gate)', zthr));
        title(ax, sprintf('ch%d gt-class%d  (%d stim)  -  responsive_count', ch, gtc, numel(stims)), ...
              'Interpreter','none');
        save_fig_local(f, fullfile(out_dir, sprintf('responsive_count_ch%d_gt%d.png', ch, gtc)));
    end
end

% ======================================================================
%  2b. TWO CANDIDATE "ALL TRIALS x ALL STIMULI" VIEWS
% ======================================================================
function val = metric_lookup(metrics, gtc, trial, stim, fname)
    val = NaN;
    idx = find([metrics.gt_class]==gtc & strcmp({metrics.trial}, trial) & [metrics.stim]==stim, 1);
    if ~isempty(idx), val = metrics(idx).m.(fname); end
end

function val = metric_lookup_row(rows, sorted_idx, sorted_stims, stim, fname)
% Same as metric_lookup, but against an already trial-filtered + sorted
% subset (rows(sorted_idx), sorted_stims), for callers doing many lookups
% against the same one trial (e.g. pairwise_metric_plots' GT crosshair).
    val = NaN;
    if isempty(rows), return; end
    k = find(sorted_stims == stim, 1);
    if ~isempty(k), val = rows(sorted_idx(k)).m.(fname); end
end

function plot_metric_trends(metrics, gt_classes, all_stims, trial_names, ref_name, out_dir, ch)
% VERSION C: one figure PER gt_class -- a line-trend PANEL PER METRIC, x =
% stimulus (YOUR order, whole list), one line per trial. Shows how each
% trial tracks/deviates from ground truth across the entire stim list.
    flds = {'zscore','mu_base','std_base','median_post','p_value_sign','p_ranksum','auroc', ...
           'resp_delta_fr_hz','mean_fr_base_hz','mean_fr_post_hz','spk_post_mean','spk_post_std', ...
           'frac_active_trials','peak_ifr','peak_ifr_lat','onset_ms','duration_ms','tons_ms', ...
           'ntrials','ifr_thr','first_spk_lat_med_ms'};
    other_trials = setdiff(trial_names, ref_name, 'stable');
    trial_order = [ref_name, other_trials];
    cmap = lines(numel(trial_order));
    for gi = 1:numel(gt_classes)
        gtc = gt_classes(gi);
        f = figure('Visible','off','Position',[80 80 1900 1400],'Color','w');
        tl = tiledlayout(f, 'flow', 'Padding','loose','TileSpacing','loose');
        title(tl, sprintf('ch%d gt-class%d  -  trend across the whole stim list (x = stim, your order)', ...
             ch, gtc), 'Interpreter','none');
        hh = [];
        for fi = 1:numel(flds)
            ax = nexttile(tl); hold(ax,'on');
            for ti = 1:numel(trial_order)
                v = arrayfun(@(s) metric_lookup(metrics, gtc, trial_order{ti}, s, flds{fi}), all_stims);
                h = plot(ax, 1:numel(all_stims), v, '-o', 'Color', cmap(ti,:), 'MarkerSize', 3, ...
                        'LineWidth', 1.2);
                if fi==1, hh(end+1) = h; end
            end
            xticks(ax, 1:numel(all_stims)); xticklabels(ax, string(all_stims));
            set(ax,'FontSize',6); xtickangle(ax,90); grid(ax,'on');
            title(ax, flds{fi}, 'Interpreter','none', 'FontSize', 9);
        end
        lg = legend(hh, trial_order, 'Interpreter','none', 'Orientation','horizontal');
        lg.Layout.Tile = 'north';
        save_fig_local(f, fullfile(out_dir, sprintf('trends_ch%d_gt%d.png', ch, gtc)));
    end
end

% ======================================================================
%  3. MU-WINDOW TRACE
% ======================================================================
function tr = mu_window_trace(ref_raw, arm_raw, ch, omap, tol, win_tol, rc, arm_trial, reject_sdnum, out_dir, name)
% Per-(stimulus,trial) comparison, working on the RAW raster rows directly --
% NOT the flattened absolute-time reconstruction (which is the wrong space
% for a window test: reconstruct_mu_times returns absolute session time,
% while the response window is relative-to-onset; mixing them silently
% matches nothing). Trial k of stim{snum} is the same physical presentation
% in every raster built from this session, so row-by-row pairing is exact,
% and gives left/right/missing "for free" without any time matching at all.
    lo = rc.tmin_median - win_tol; hi = rc.tmax_median + win_tol;
    cf = sprintf('chan%d', ch);
    ref_mu = ref_raw.rasters.(cf).mu.stim;
    arm_mu = arm_raw.rasters.(cf).mu.stim;

    n_ref = 0; n_matched = 0; n_left = 0; n_right = 0;
    miss_snum = []; miss_k = []; miss_t = []; miss_abs = [];
    for snum = 1:min(numel(ref_mu), numel(arm_mu))
        rtr = ref_mu{snum}; atr = arm_mu{snum};
        if isempty(rtr) || isempty(atr), continue; end
        if ~iscell(rtr), rtr = num2cell(rtr,2); end
        if ~iscell(atr), atr = num2cell(atr,2); end
        if ~isKey(omap, snum), continue; end
        O = omap(snum);
        ntr = min([numel(rtr), numel(atr), numel(O)]);
        for k = 1:ntr
            rv = rtr{k}(:).'; rv = rv(rv>=lo & rv<=hi & rv~=9999 & rv~=10000);
            if isempty(rv), continue; end
            av = atr{k}(:).'; av = av(av~=9999 & av~=10000);
            for t0 = rv
                n_ref = n_ref + 1;
                if any(abs(av - t0) <= tol), n_matched = n_matched + 1; continue; end
                cand = av(av>=t0-win_tol & av<=t0+win_tol);
                if any(cand < t0), n_left = n_left + 1;
                elseif any(cand > t0), n_right = n_right + 1;
                else
                    miss_snum(end+1) = snum; miss_k(end+1) = k; miss_t(end+1) = t0;
                    miss_abs(end+1) = O(k) + t0;
                end
            end
        end
    end
    n_missing = numel(miss_abs);

    % trace: what class (if any) does each truly-missing spike belong to in
    % this trial's own clustering, and is it quarantined -- via ABSOLUTE time.
    status = repmat({'unknown'}, n_missing, 1);
    if ~isempty(arm_trial.times_file) && exist(arm_trial.times_file, 'file')
        TF = load(arm_trial.times_file);
        gm2 = nearest_match(miss_abs(:), TF.cluster_class(:,2), tol);
        for ii = 1:n_missing
            if gm2(ii) > 0
                cl = TF.cluster_class(gm2(ii),1);
                if cl == 0, status{ii} = 'in-arm class0';
                else, status{ii} = sprintf('in-arm class%d', cl); end
            end
        end
    end
    if ~isempty(arm_trial.spikes_file) && exist(arm_trial.spikes_file,'file')
        SPK = load(arm_trial.spikes_file);
        if isfield(SPK,'index_all')
            gm3 = nearest_match(miss_abs(:), double(SPK.index_all(:)), tol);
            qmasks = intersect({'mask_nonart','mask_non_quarantine','mask_non_refract'}, fieldnames(SPK));
            for ii = 1:n_missing
                if strcmp(status{ii},'unknown') && gm3(ii)>0
                    failed = {};
                    for qm = qmasks
                        if ~logical(SPK.(qm{1})(gm3(ii))), failed{end+1} = qm{1}; end
                    end
                    if ~isempty(failed), status{ii} = ['quarantined: ' strjoin(failed,'+')];
                    else, status{ii} = 'detected, not in times file (?)'; end
                elseif strcmp(status{ii},'unknown') && gm3(ii)==0
                    status{ii} = 'not detected at all';
                end
            end
        end
    end
    status_tbl = tabulate_status(status);

    tr = struct('trial', name, 'n_ref', n_ref, 'n_matched', n_matched, ...
               'n_left', n_left, 'n_right', n_right, 'n_missing', n_missing, ...
               'missing_status', {status_tbl});
    if ~isempty(status_tbl), writetable(status_tbl, fullfile(out_dir, sprintf('mu_trace_ch%d_%s.csv', ch, name))); end

    f = figure('Visible','off','Position',[100 100 900 460],'Color','w');
    tl = tiledlayout(f,1,2,'Padding','loose','TileSpacing','loose');
    ax = nexttile(tl);
    bar(ax, categorical({'matched','left','right','missing'}), [n_matched n_left n_right n_missing]);
    grid(ax,'on'); ylabel(ax,'# ref mu spikes');
    title(ax, sprintf('ch%d %s vs ref (n_ref=%d)', ch, name, n_ref), 'Interpreter','none');
    ax = nexttile(tl);
    if ~isempty(status_tbl)
        bar(ax, categorical(status_tbl.status), status_tbl.n); grid(ax,'on');
        set(ax,'TickLabelInterpreter','none'); xtickangle(ax,30);
        ylabel(ax,'# missing spikes'); title(ax,'where did the missing ones go?');
    end
    save_fig_local(f, fullfile(out_dir, sprintf('mu_trace_ch%d_%s.png', ch, name)));
end

function T = tabulate_status(status)
    if isempty(status), T = table(); return; end
    [u, ~, ic] = unique(status);
    n = accumarray(ic, 1);
    T = table(u(:), n(:), 'VariableNames', {'status','n'});
    T = sortrows(T, 'n', 'descend');
end

% ======================================================================
%  WAVEFORM OVERLAY (adapted from validate_recluster_vs_handselect's
%  waveform_overlay -- same plot design: mean +/- std fill, one line per
%  trial, GT in black -- reshaped for our trials(name/grapes_file/
%  times_file/spikes_file) struct instead of its arm_info)
% ======================================================================
function waveform_overlay_plot(ch, gtc, ref, trials, ref_i, armmap, ref_spikes_file, match_tol_ms, out_dir)
    if ~exist(out_dir,'dir'), mkdir(out_dir); end
    f = figure('Visible','off','Position',[100 100 900 520],'Color','w');
    ax = axes(f); hold(ax,'on');
    cmap = lines(numel(trials)+1);
    leg = {}; hh = []; plotted = false;

    % Same full-vs-windowed rule as membership/class_match: per arm, use the
    % times file's full whole-session class (matched against the raw spikes
    % file for waveforms -- never trusting times_file's own 'spikes' field
    % directly) only when BOTH GT's and that arm's times-vs-grapes counts
    % agree; otherwise windowed for that side. GT and each arm are decided
    % independently here (unlike membership/class_match, this plot has no
    % single shared "scope" -- each line is its own trial), each line's
    % legend says which scope it used.
    [gt_ok,~,~] = check_class_count_agreement(ref.times_file, ref.grapes_file, ch, gtc);

    % ---- GT: match its spike times (full or windowed) to a REAL spikes file ----
    if ~isempty(ref_spikes_file) && exist(ref_spikes_file,'file')
        S = load(ref_spikes_file);
        if isfield(S,'spikes_all') && isfield(S,'index_all')
            SA = S.spikes_all; IA = double(S.index_all(:));
        elseif isfield(S,'spikes') && isfield(S,'index')
            SA = S.spikes; IA = double(S.index(:));
        else
            SA = []; IA = [];
        end
        if ~isempty(SA)
            if gt_ok
                gt_t = full_class_times_from_times_file(ref.times_file, gtc); gt_scope = 'full';
            else
                gt_t = ref.cc(ref.cc(:,1)==gtc, 2); gt_scope = 'win';
            end
            idx = nearest_match(gt_t, IA, match_tol_ms);
            idx = idx(idx>0);
            w = SA(idx,:);
            if ~isempty(w)
                mu = mean(w,1); sd = std(w,0,1); xx = 1:numel(mu);
                fill(ax, [xx fliplr(xx)], [mu+sd fliplr(mu-sd)], [0 0 0], ...
                    'FaceAlpha',0.12,'EdgeColor','none','HandleVisibility','off');
                h = plot(ax, xx, mu, '-', 'Color',[0 0 0],'LineWidth',2.5);
                hh(end+1) = h;
                leg{end+1} = sprintf('handselect (gt c%d, %s, n=%d)', gtc, gt_scope, size(w,1));
                plotted = true;
            end
        end
    end

    for i = 1:numel(trials)
        if i == ref_i, continue; end
        sf = trials(i).spikes_file;
        if isempty(sf) || ~exist(sf,'file') || ~isKey(armmap, trials(i).name), continue; end
        acl = armmap(trials(i).name);
        if isnan(acl), continue; end
        S = load(sf);
        if isfield(S,'spikes_all') && isfield(S,'index_all')
            SA = S.spikes_all; IA = double(S.index_all(:));
        elseif isfield(S,'spikes') && isfield(S,'index')
            SA = S.spikes; IA = double(S.index(:));
        else
            SA = []; IA = [];
        end
        if isempty(SA), continue; end
        [arm_ok,~,~] = check_class_count_agreement(trials(i).times_file, trials(i).grapes_file, ch, acl);
        if gt_ok && arm_ok
            arm_t = full_class_times_from_times_file(trials(i).times_file, acl); arm_scope = 'full';
        else
            arm_t = trials(i).cc(trials(i).cc(:,1)==acl, 2); arm_scope = 'win';
        end
        idx = nearest_match(arm_t, IA, match_tol_ms);
        idx = idx(idx>0);
        w = SA(idx,:);
        if isempty(w), continue; end
        mu = mean(w,1); sd = std(w,0,1); xx = 1:numel(mu);
        cc = cmap(i,:);
        fill(ax, [xx fliplr(xx)], [mu+sd fliplr(mu-sd)], cc, ...
            'FaceAlpha',0.12,'EdgeColor','none','HandleVisibility','off');
        h = plot(ax, xx, mu, '-', 'Color', cc, 'LineWidth', 1.2);
        hh(end+1) = h;
        leg{end+1} = sprintf('%s (c%d, %s, n=%d)', trials(i).name, acl, arm_scope, size(w,1));
        plotted = true;
    end
    if ~plotted, close(f); return; end
    hold(ax,'off'); grid(ax,'on');
    legend(ax, hh, leg, 'Interpreter','none','Location','best');
    title(ax, sprintf('ch%d  ground-truth class %d  mean waveform +/- std', ch, gtc), 'Interpreter','none');
    xlabel(ax,'sample'); ylabel(ax,'amplitude');
    save_fig_local(f, fullfile(out_dir, sprintf('waveform_overlay_ch%d_gt%d.png', ch, gtc)));
end

% ======================================================================
%  SHARED HELPERS
% ======================================================================
function gm = nearest_match(gt_t, ar_t, tol)
    gm = zeros(numel(gt_t), 1);
    if isempty(ar_t), return; end
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

function save_fig_local(f, out_png)
    set(f, 'Color', 'w');
    % defensive: recreate the output folder right before writing -- a run
    % this long (dozens of plot_grapes_as_online calls, each pre-caching
    % ~350 images) hit a real crash here once (folder gone by the 3rd trial,
    % cause unconfirmed -- possibly a network-mount hiccup on a long run);
    % this makes save_fig_local self-healing regardless of cause instead of
    % taking the whole run down on one missing folder.
    od = fileparts(out_png);
    if ~isempty(od) && ~exist(od, 'dir')
        mkdir(od);
    end
    try
        print(f, out_png, '-dpng', '-r130');
    catch
        try
            exportgraphics(f, out_png, 'Resolution', 130, 'BackgroundColor', 'white');
        catch ME
            warning('save_fig_local: failed to write %s: %s', out_png, ME.message);
        end
    end
    close(f);
end
