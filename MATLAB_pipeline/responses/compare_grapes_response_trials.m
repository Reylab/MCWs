function cmp = compare_grapes_response_trials(varargin)
% COMPARE_GRAPES_RESPONSE_TRIALS  Compare one channel's stimulus responses
% across several processing "trials" (mask variants, sorting settings, ...).
%
% Each trial is a folder in which characterize_grapes_responses.m was run,
% producing a grapes_response_metrics_ch<NN>.mat. This script lines those
% up and, for a fixed channel + unit class, builds stim x trial matrices of
% every response metric so a single stimulus response can be tracked across
% trials.
%
% Name/value parameters
%   'trials'     : how to choose the trial files (need >= 2). One of:
%                  - {} (default): if a display is available, opens a file
%                    picker; keep selecting files (multi-select, across
%                    folders) and Cancel when done - the files gathered so
%                    far are the comparison set. Pick nothing / no display
%                    -> falls back to every grapes_response_metrics_*.mat
%                    under pwd and its immediate subfolders.
%                  - 'ui'  : force the picker even if 'trials' was given.
%                  - 'auto': skip the picker, use the pwd glob above.
%                  - cell array, one entry per trial: each a
%                    grapes_response_metrics_*.mat file, or a folder holding
%                    one (disambiguated with 'channel').
%   'labels'     : cell array of trial names (default: the folder one level
%                  below meta.session_root for each trial - i.e. the trial
%                  folder under the session root - made unique; falls back
%                  to the metrics file's parent folder name).
%   'channel'    : [] (infer, must be unique across files) or channel number.
%   'class'      : 'mu' (default) or 'classN'.
%   'grapes_tag' : 'auto' (default) or the raster flavour key to read
%                  ('grapes','grapes_blanks','grapes_blanks_circ',...).
%                  'auto' picks the flavour present in every trial,
%                  preferring grapes_blanks_circ > grapes_blanks > grapes.
%   'stim_list'  : 'all' (default, union of stims over trials) or a numeric
%                  vector of stimulus numbers.
%   'metrics'    : 'all' (default) or a cellstr subset of metric columns.
%   'reference'  : [] (default) or a trial index / label. When set, also
%                  emits cmp.delta.<metric> = metric - reference metric.
%                  In 'rescue' mode it defaults to 1 (pre_rescue).
%   'mode'       : 'auto' (default) | 'trials' | 'rescue'.
%                  'trials' - compare the same flavour across >= 2 trial
%                    folders (the original behaviour).
%                  'rescue' - compare pre- vs post-rescue WITHIN one metrics
%                    file, reading the self-contained <flavour>_rescue block
%                    written by do_structure_rescue_BCM_online3 +
%                    characterize_grapes_responses: column 1 = class<N>_prerescue,
%                    column 2 = class<N>. (Older files without class<N>_prerescue
%                    fall back to the plain <flavour> block as the baseline -
%                    only valid if grapes.mat was not regenerated after rescue.)
%                    Labels default to {'pre_rescue','post_rescue'}; reference
%                    defaults to 1 so cmp.delta = post - pre. For class = classN
%                    it also fills cmp.rescued_only with the class<N>_rescue rows.
%                  'auto' -> 'trials' when >= 2 files are found, else 'rescue'.
%   'expand_rescue' : false (default). 'trials' mode only. When true, each
%                  trial whose metrics file also carries a <flavour>_rescue
%                  block gets a SECOND column ('<label>_rescue') alongside its
%                  plain '<label>' column; trials that only have the plain
%                  flavour contribute their one column as usual. Use this to
%                  line pre- and post-rescue metrics up across a mixed set of
%                  trials. Pass the BASE flavour to 'grapes_tag' (e.g.
%                  'grapes'), not 'grapes_rescue'.
%   'prefer_rescue' : false (default). 'trials' mode only. ONE column per
%                  trial (unlike 'expand_rescue'): a trial whose metrics file
%                  carries a <flavour>_rescue block is read from THAT block
%                  (post-rescue), the others from the plain <flavour> block.
%                  Rescued columns get a ' +rescue' label suffix. Use this to
%                  put e.g. two rescued trials and one un-rescued trial side by
%                  side, each as a single column. Pass the BASE flavour to
%                  'grapes_tag'. Mutually exclusive with 'expand_rescue'.
%   'outfile'    : default 'compare_response_ch<NN>_<class>.mat' in pwd.
%   'save_mat'   : true (default) - write outfile.
%   'make_plots' : true (default) - write PNGs to 'plot_dir'.
%   'plot_dir'   : default 'compare_response_ch<NN>_<class>_plots' in pwd.
%   'plot_metrics' : metrics to get change-line + heatmap plots. Default
%                  {'zscore','auroc','resp_delta_fr_hz','p_value_sign',
%                   'peak_ifr','onset','frac_active_trials'}.
%   'show_images': true (default) - per-stimulus panel (image + mini change
%                  plots) saved under <plot_dir>/per_stimulus/. Auto-skipped
%                  when nstim > 'max_plot_stims'.
%   'max_plot_stims' : 40 (default). When more stimuli than this are compared
%                  the per-stimulus line plots and stim x trial heatmaps
%                  (unreadable at that size) are replaced by one compact
%                  scatter_<metric>.png per metric: reference-trial value on x,
%                  each other trial on y, one point per stimulus.
%   'pics_dir'   : '' (default, taken from a trial's meta.pics_dir) or an
%                  explicit folder of stimulus images named like stim_name.
%
% Output (also saved unless save_mat=false)
%   cmp.trials       - table: label, file, dir, flavour, class_read, channel, n_stim
%   cmp.channel/class/grapes_tag/mode
%   cmp.rescued_only - (rescue mode, classN) metrics rows for classN_rescue
%   cmp.stim_number  - [nstim x 1]   stimulus numbers (row order of matrices)
%   cmp.stim_name    - [nstim x 1] string
%   cmp.metrics_list - cellstr of metric names
%   cmp.wide.<metric>- [nstim x ntrial] double  (NaN where a trial lacks it)
%   cmp.delta.<metric> - [nstim x ntrial] vs reference trial (if 'reference')
%   cmp.long         - stacked table (trial,channel,class,stim_number,
%                      stim_name,<metrics>) for all classes present
%   cmp.summary      - per-trial table: n_stim_present, n_responsive,
%                      mean/median zscore, mean auroc, mean resp_delta_fr_hz,
%                      frac_good_lat
%   cmp.pics_dir / cmp.plot_dir
%
% Plots written to plot_dir (make_plots=true):
%   summary.png                    per-trial summary bar charts (x = column #,
%                                  key in the x-label)
%   nstim <= max_plot_stims:
%     change_<metric>.png          metric vs trial, one line per stimulus
%     heatmap_<metric>.png         stim x trial image (+ heatmap_delta_* if a
%                                  reference trial was given)
%     per_stimulus/stim_<n>.png    stimulus image + per-metric bars (show_images)
%   nstim >  max_plot_stims:
%     scatter_<metric>.png         reference trial (x) vs each other trial (y),
%                                  one point per stimulus, diagonal = no change

    ipr = inputParser;
    addParameter(ipr, 'trials', {});
    addParameter(ipr, 'labels', {});
    addParameter(ipr, 'channel', []);
    addParameter(ipr, 'class', 'mu');
    addParameter(ipr, 'grapes_tag', 'auto');
    addParameter(ipr, 'stim_list', 'all');
    addParameter(ipr, 'metrics', 'all');
    addParameter(ipr, 'reference', []);
    addParameter(ipr, 'mode', 'auto');
    addParameter(ipr, 'expand_rescue', false, @islogical);
    addParameter(ipr, 'prefer_rescue', false, @islogical);
    addParameter(ipr, 'outfile', '');
    addParameter(ipr, 'save_mat', true, @islogical);
    addParameter(ipr, 'make_plots', true, @islogical);
    addParameter(ipr, 'plot_dir', '');
    addParameter(ipr, 'plot_metrics', {'zscore','auroc','resp_delta_fr_hz', ...
        'p_value_sign','peak_ifr','onset','frac_active_trials'});
    addParameter(ipr, 'show_images', true, @islogical);
    addParameter(ipr, 'max_plot_stims', 40, @(x) isnumeric(x) && isscalar(x) && x >= 1);
    addParameter(ipr, 'pics_dir', '');
    parse(ipr, varargin{:});
    p = ipr.Results;

    cls = char(p.class);

    % ---- locate the per-trial metrics files ----------------------------
    use_ui = (ischar(p.trials) || isstring(p.trials)) && strcmpi(p.trials, 'ui');
    use_glob = (ischar(p.trials) || isstring(p.trials)) && strcmpi(p.trials, 'auto');
    if use_ui || (isempty(p.trials) && ~use_glob)
        files = pick_trial_files_ui(pwd);
        if isempty(files)
            fprintf(['No files picked - falling back to grapes_response_metrics_*.mat ' ...
                     'under %s\n'], pwd);
            files = resolve_trial_files({}, p.channel);
        end
    elseif use_glob || isempty(p.trials)
        files = resolve_trial_files({}, p.channel);
    else
        files = resolve_trial_files(p.trials, p.channel);
    end
    mode = lower(char(p.mode));
    if strcmp(mode, 'auto')
        if numel(files) >= 2, mode = 'trials'; else, mode = 'rescue'; end
    end
    if isempty(files)
        error('compare_grapes_response_trials:noTrials', ...
              'no grapes_response_metrics_*.mat found.');
    end
    if strcmp(mode, 'trials') && numel(files) < 2
        error('compare_grapes_response_trials:fewTrials', ...
              ['''trials'' mode needs >= 2 trial files, got %d. Pick more, pass ' ...
               '''trials'' explicitly, run from a folder with >= 2 ' ...
               'grapes_response_metrics_*.mat, or use ''mode'',''rescue'' to ' ...
               'compare pre/post-rescue within one file.'], numel(files));
    end
    if strcmp(mode, 'rescue') && numel(files) > 1
        warning(['rescue mode compares flavours within ONE metrics file; ' ...
                 'using %s'], files{1});
        files = files(1);
    end
    fprintf('Mode: %s.  Files:\n', mode);
    fprintf('  %s\n', files{:});

    % ---- load every trial -------------------------------------------
    S = cell(numel(files), 1);
    tag_sets = cell(numel(files), 1);
    pics_dir = char(p.pics_dir);
    for i = 1:numel(files)
        S{i} = load(files{i});
        tag_sets{i} = struct_metric_tags(S{i});
        if isempty(tag_sets{i})
            error('%s has no <tag>.metrics table - is it a characterize_grapes_responses output?', files{i});
        end
        if isempty(pics_dir) && isfield(S{i}, 'meta') && isfield(S{i}.meta, 'pics_dir') ...
                && exist(S{i}.meta.pics_dir, 'dir')
            pics_dir = S{i}.meta.pics_dir;
        end
    end

    % ---- labels: the folder one level below meta.session_root --------
    if ~isempty(p.labels)
        labels = cellstr(p.labels);
    else
        labels = cell(size(files));
        for i = 1:numel(files)
            meta = [];
            if isfield(S{i}, 'meta'), meta = S{i}.meta; end
            labels{i} = label_from_meta(meta, files{i});
        end
        labels = matlab.lang.makeUniqueStrings(labels);
    end

    % ---- build the comparison columns -----------------------------
    %   col_S{i}   - loaded metrics struct for column i
    %   col_tag{i} - which out.<tag> block to read
    %   col_class{i} - which value of the .class column to filter to
    if strcmp(mode, 'rescue')
        % Locate the "<flavour>_rescue" metrics block written by
        % do_structure_rescue_BCM_online3 + characterize_grapes_responses.
        if strcmpi(p.grapes_tag, 'auto')
            rcand = tag_sets{1}(endsWith(tag_sets{1}, '_rescue'));
            rpref = {'grapes_blanks_circ_rescue','grapes_blanks_rescue','grapes_rescue'};
            rtag  = '';
            for k = 1:numel(rpref)
                if ismember(rpref{k}, rcand), rtag = rpref{k}; break; end
            end
            if isempty(rtag) && ~isempty(rcand), rtag = rcand{1}; end
        else
            gt = char(p.grapes_tag);
            if endsWith(gt, '_rescue'), rtag = gt; else, rtag = [gt '_rescue']; end
        end
        if isempty(rtag) || ~ismember(rtag, tag_sets{1})
            error(['rescue mode: no "*_rescue" metrics block in %s. Run ' ...
                   'do_structure_rescue_BCM_online3, then characterize_grapes_responses.'], files{1});
        end
        gtag    = erase(rtag, '_rescue');              % representative base name
        Trej    = S{1}.(rtag).metrics;
        pre_cls = [cls '_prerescue'];
        have_pre_in_file = any(strcmp(string(Trej.class), pre_cls));

        ntr      = 2;
        col_S    = {S{1}, S{1}};
        col_file = {files{1}, files{1}};
        if have_pre_in_file
            % self-contained: pre and post both live in the _rescue block
            col_tag   = {rtag, rtag};
            col_class = {pre_cls, cls};
        elseif ismember(gtag, tag_sets{1}) && any(strcmp(string(S{1}.(gtag).metrics.class), cls))
            % fall back to the separate plain "<flavour>" block as the baseline
            warning(['rescue mode: "%s" not in the %s block - using the plain "%s" ' ...
                     'block as the pre-rescue baseline. Rebuild grapes_rescue.mat with ' ...
                     'the current do_structure_rescue_BCM_online3, or make sure ' ...
                     'grapes.mat was NOT regenerated after rescue_spikes.'], pre_cls, rtag, gtag);
            col_tag   = {gtag, rtag};
            col_class = {cls, cls};
        elseif any(strcmp(string(Trej.class), cls))
            % no pre-rescue baseline for this unit (e.g. class='mu'): rescue
            % does not modify it, so both columns are the same data.
            warning(['rescue mode: no pre-rescue baseline for "%s" - rescue_spikes ' ...
                     'does not modify this unit, so pre and post columns are identical.'], cls);
            col_tag   = {rtag, rtag};
            col_class = {cls, cls};
        else
            error(['rescue mode: %s has no "%s" rows, no plain "%s" block, and no ' ...
                   '"%s" rows in %s.'], files{1}, pre_cls, gtag, cls, rtag);
        end
        if ~isempty(p.labels)
            labels = cellstr(p.labels);
        else
            labels = {'pre_rescue', 'post_rescue'};
        end
        assert(numel(labels) == 2, 'rescue mode needs exactly 2 labels, got %d', numel(labels));
    else
        assert(numel(labels) == numel(files), ...
               'labels count (%d) ~= trial count (%d)', numel(labels), numel(files));
        % base flavour common to all trial files
        if strcmpi(p.grapes_tag, 'auto')
            common = tag_sets{1};
            for i = 2:numel(tag_sets), common = intersect(common, tag_sets{i}); end
            common = common(~endsWith(common, {'_rescue','_prerescue'}));
            if isempty(common)
                error('trials share no common grapes flavour; tags: %s', ...
                      strjoin(cellfun(@(c) strjoin(c, ','), tag_sets, 'UniformOutput', false), ' | '));
            end
            pref = {'grapes_blanks_circ','grapes_blanks','grapes'};
            gtag = '';
            for k = 1:numel(pref)
                if ismember(pref{k}, common), gtag = pref{k}; break; end
            end
            if isempty(gtag), gtag = common{1}; end
        else
            gtag = char(p.grapes_tag);
            for i = 1:numel(tag_sets)
                ok = ismember(gtag, tag_sets{i}) || ...
                     (p.prefer_rescue && ismember([gtag '_rescue'], tag_sets{i}));
                if ~ok
                    error('trial "%s" (%s) has no flavour "%s"%s (has: %s)', ...
                          labels{i}, files{i}, gtag, ...
                          repmat(' or its _rescue variant', 1, p.prefer_rescue), ...
                          strjoin(tag_sets{i}, ', '));
                end
            end
        end
        if p.expand_rescue && p.prefer_rescue
            error(['compare_grapes_response_trials: ''expand_rescue'' and ' ...
                   '''prefer_rescue'' are mutually exclusive.']);
        end
        col_S = {}; col_tag = {}; col_file = {}; col_lab = {}; col_class = {};
        n_pref = 0;
        for i = 1:numel(files)
            has_r = ismember([gtag '_rescue'], tag_sets{i});
            if p.prefer_rescue && has_r
                col_S{end+1}     = S{i};
                col_tag{end+1}   = [gtag '_rescue'];
                col_file{end+1}  = files{i};
                col_lab{end+1}   = [labels{i} ' +rescue'];
                col_class{end+1} = cls;
                n_pref = n_pref + 1;
            else
                col_S{end+1}     = S{i};
                col_tag{end+1}   = gtag;
                col_file{end+1}  = files{i};
                col_lab{end+1}   = labels{i};
                col_class{end+1} = cls;
            end
            if p.expand_rescue && has_r
                col_S{end+1}     = S{i};
                col_tag{end+1}   = [gtag '_rescue'];
                col_file{end+1}  = files{i};
                col_lab{end+1}   = [labels{i} '_rescue'];
                col_class{end+1} = cls;
            end
        end
        labels = col_lab;
        ntr    = numel(col_S);
        if p.expand_rescue
            fprintf('expand_rescue: %d trial file(s) -> %d column(s) (%d with _rescue)\n', ...
                    numel(files), ntr, ntr - numel(files));
        elseif p.prefer_rescue
            fprintf('prefer_rescue: %d/%d trial column(s) read from a *_rescue block.\n', ...
                    n_pref, ntr);
        end
    end
    fprintf('Mode %s: base flavour "%s", class "%s", %d column(s): %s.\n', ...
            mode, gtag, cls, ntr, strjoin(labels, ', '));

    % ---- pull the per-column metrics rows for this channel+class ----
    rows      = cell(ntr, 1);
    chan_seen = [];
    for i = 1:ntr
        T = col_S{i}.(col_tag{i}).metrics;
        m = strcmp(string(T.class), col_class{i});
        if ~isempty(p.channel)
            m = m & (T.channel_num == p.channel);
        end
        Ti = T(m, :);
        if isempty(Ti)
            warning('column "%s": no rows for class %s (channel %s)', ...
                    labels{i}, col_class{i}, mat2str(p.channel));
        else
            chan_seen = unique([chan_seen; unique(Ti.channel_num)]);
        end
        rows{i} = Ti;
    end
    if isempty(chan_seen)
        error('No matching rows in any column for class "%s".', cls);
    end
    if ~isscalar(chan_seen)
        error('Multiple channels present (%s). Pass ''channel'' to pick one.', ...
              mat2str(chan_seen(:)'));
    end
    channel = chan_seen;

    % ---- rescued-only block (class<N>_rescue), rescue mode only -----
    rescued_only = table();
    if strcmp(mode, 'rescue') && ~isempty(regexp(cls, '^class\d+$', 'once'))
        Tr = col_S{2}.(col_tag{2}).metrics;
        mr = strcmp(string(Tr.class), [cls '_rescue']) & (Tr.channel_num == channel);
        rescued_only = Tr(mr, :);
        fprintf('rescued-only rows for %s_rescue: %d\n', cls, height(rescued_only));
    end

    % ---- stimulus set (row order) -----------------------------------
    if ischar(p.stim_list) || isstring(p.stim_list)
        stim_number = [];
        for i = 1:numel(rows), stim_number = union(stim_number, rows{i}.stim_number); end
    else
        stim_number = unique(double(p.stim_list(:)));
    end
    stim_number = stim_number(:);
    nstim = numel(stim_number);

    % stim names: first non-empty across trials
    stim_name = strings(nstim, 1);
    for i = 1:numel(rows)
        [tf, loc] = ismember(stim_number, rows{i}.stim_number);
        nm = strings(nstim, 1);
        nm(tf) = string(rows{i}.stim_name(loc(tf)));
        fill = stim_name == "" & nm ~= "";
        stim_name(fill) = nm(fill);
    end

    % ---- metric columns -------------------------------------------
    example = rows{find(~cellfun(@isempty, rows), 1)};
    id_cols = {'channel','ch_label','class','stim_name'};
    all_metric_cols = setdiff(example.Properties.VariableNames, [id_cols, {'IFR'}], 'stable');
    keep = false(size(all_metric_cols));
    for k = 1:numel(all_metric_cols)
        v = example.(all_metric_cols{k});
        keep(k) = isnumeric(v) || islogical(v);
    end
    all_metric_cols = all_metric_cols(keep);
    if ischar(p.metrics) || isstring(p.metrics)
        if strcmpi(p.metrics, 'all')
            metric_cols = all_metric_cols;
        else
            metric_cols = intersect(cellstr(p.metrics), all_metric_cols, 'stable');
        end
    else
        metric_cols = intersect(cellstr(p.metrics), all_metric_cols, 'stable');
    end

    % ---- wide matrices: stim x column -----------------------------
    wide = struct();
    for k = 1:numel(metric_cols)
        M = nan(nstim, ntr);
        for i = 1:ntr
            if isempty(rows{i}), continue; end
            [tf, loc] = ismember(stim_number, rows{i}.stim_number);
            col = double(rows{i}.(metric_cols{k}));
            M(tf, i) = col(loc(tf));
        end
        wide.(metric_cols{k}) = M;
    end

    % ---- long table (all classes, for completeness) ---------------
    % de-dup identical (struct,tag) columns (rescue mode reads one block twice)
    long = table();
    seen_src = strings(0,1);
    for i = 1:ntr
        src_key = string(col_tag{i}) + "@" + string(col_file{i});
        if any(seen_src == src_key), continue; end
        seen_src(end+1) = src_key;
        T = col_S{i}.(col_tag{i}).metrics;
        m = (T.channel_num == channel);
        if ~(ischar(p.stim_list) || isstring(p.stim_list))
            m = m & ismember(T.stim_number, stim_number);
        end
        Ti = T(m, :);
        if ismember('IFR', Ti.Properties.VariableNames)
            Ti.IFR = [];
        end
        if isempty(Ti), continue; end
        if strcmp(mode, 'rescue')
            trial_val = string(col_tag{i});
        else
            trial_val = string(labels{i});
        end
        Ti = addvars(Ti, repmat(trial_val, height(Ti), 1), ...
                     'Before', 1, 'NewVariableNames', 'trial');
        long = [long; Ti];
    end

    % ---- per-column summary ------------------------------------
    sm_label = strings(ntr,1);
    sm_np = zeros(ntr,1); sm_nr = zeros(ntr,1);
    sm_mz = nan(ntr,1);   sm_medz = nan(ntr,1);
    sm_auroc = nan(ntr,1); sm_dfr = nan(ntr,1);
    sm_gl = nan(ntr,1);
    for i = 1:ntr
        sm_label(i) = string(labels{i});
        Ti = rows{i};
        if isempty(Ti), continue; end
        inset = ismember(Ti.stim_number, stim_number);
        Ti = Ti(inset, :);
        sm_np(i) = height(Ti);
        sm_nr(i) = sum(Ti.min_spk_test & Ti.p_test & Ti.good_lat);
        sm_mz(i) = mean(Ti.zscore, 'omitnan');
        sm_medz(i) = median(Ti.zscore, 'omitnan');
        if ismember('auroc', Ti.Properties.VariableNames)
            sm_auroc(i) = mean(Ti.auroc, 'omitnan');
        end
        if ismember('resp_delta_fr_hz', Ti.Properties.VariableNames)
            sm_dfr(i) = mean(Ti.resp_delta_fr_hz, 'omitnan');
        end
        sm_gl(i) = mean(double(Ti.good_lat), 'omitnan');
    end
    summary = table(sm_label, sm_np, sm_nr, sm_mz, sm_medz, sm_auroc, sm_dfr, sm_gl, ...
        'VariableNames', {'trial','n_stim_present','n_responsive','mean_zscore', ...
                          'median_zscore','mean_auroc','mean_resp_delta_fr_hz','frac_good_lat'});

    % ---- reference deltas ------------------------------------
    delta = struct();
    ref_idx = [];
    ref_req = p.reference;
    if isempty(ref_req) && strcmp(mode, 'rescue')
        ref_req = 1;   % delta.<metric> = post_rescue - pre_rescue
    end
    if ~isempty(ref_req)
        if isnumeric(ref_req)
            ref_idx = ref_req;
        else
            ref_idx = find(strcmp(labels, char(ref_req)), 1);
        end
        if isempty(ref_idx) || ref_idx < 1 || ref_idx > ntr
            warning('reference "%s" not matched - skipping deltas', mat2str(ref_req));
            ref_idx = [];
        else
            for k = 1:numel(metric_cols)
                M = wide.(metric_cols{k});
                delta.(metric_cols{k}) = M - M(:, ref_idx);
            end
        end
    end

    % ---- assemble output -----------------------------------
    [tf_dir, ~] = cellfun(@fileparts, col_file, 'UniformOutput', false);
    trials_tbl = table(string(labels(:)), string(col_file(:)), string(tf_dir(:)), ...
                       string(col_tag(:)), string(col_class(:)), repmat(channel, ntr, 1), ...
                       cellfun(@(r) height(r), rows), ...
        'VariableNames', {'label','file','dir','flavour','class_read','channel','n_stim'});

    cmp = struct();
    cmp.created      = char(datetime('now','Format','yyyy-MM-dd HH:mm:ss'));
    cmp.mode         = mode;
    cmp.expand_rescue = p.expand_rescue && strcmp(mode, 'trials');
    cmp.prefer_rescue = p.prefer_rescue && strcmp(mode, 'trials');
    cmp.channel      = channel;
    cmp.class        = cls;
    cmp.grapes_tag   = gtag;
    cmp.reference    = ref_idx;
    cmp.trials       = trials_tbl;
    cmp.stim_number  = stim_number;
    cmp.stim_name    = stim_name;
    cmp.metrics_list = metric_cols(:)';
    cmp.wide         = wide;
    cmp.delta        = delta;
    cmp.long         = long;
    cmp.summary      = summary;
    cmp.pics_dir     = pics_dir;
    if strcmp(mode, 'rescue')
        cmp.rescued_only = rescued_only;
    end

    % ---- console preview ----------------------------------
    disp(summary);
    if ismember('zscore', metric_cols) && nstim <= 40
        Z = array2table(wide.zscore, 'VariableNames', ...
            matlab.lang.makeValidName(labels), ...
            'RowNames', compose('stim%d_%s', stim_number, stim_name));
        fprintf('\nzscore  (stim x trial):\n');
        disp(Z);
    end

    % ---- save --------------------------------------------
    if p.save_mat
        outfile = p.outfile;
        if isempty(outfile)
            if strcmp(mode, 'rescue')
                outfile = sprintf('compare_rescue_ch%d_%s.mat', channel, cls);
            else
                outfile = sprintf('compare_response_ch%d_%s.mat', channel, cls);
            end
        end
        save(outfile, '-struct', 'cmp', '-v7');
        if isempty(fileparts(outfile))
            fprintf('\nSaved %s\n', fullfile(pwd, outfile));
        else
            fprintf('\nSaved %s\n', outfile);
        end
    end

    % ---- plots ------------------------------------------
    if p.make_plots
        plot_dir = p.plot_dir;
        if isempty(plot_dir)
            if strcmp(mode, 'rescue')
                plot_dir = sprintf('compare_rescue_ch%d_%s_plots', channel, cls);
            else
                plot_dir = sprintf('compare_response_ch%d_%s_plots', channel, cls);
            end
        end
        try
            make_comparison_plots(cmp, labels, plot_dir, ...
                                  cellstr(p.plot_metrics), p.show_images, pics_dir, ...
                                  p.max_plot_stims);
            fprintf('Plots -> %s\n', plot_dir);
            cmp.plot_dir = plot_dir;
        catch ME
            warning('compare_grapes_response_trials:plotFailed', ...
                    'plotting failed: %s', getReport(ME));
        end
    end
end

% ======================================================================
function files = pick_trial_files_ui(start_dir)
% Repeatedly open a multi-select file picker; accumulate the chosen files
% across folders. Cancel finishes selection. Returns {} if nothing picked
% or if no GUI is available (caller then falls back to a glob).
    files = {};
    if ~can_show_dialogs()
        return
    end
    d = start_dir;
    while true
        [fn, pth] = uigetfile( ...
            {'grapes_response_metrics_*.mat', 'trial metrics (grapes_response_metrics_*.mat)'; ...
             '*.mat', 'MAT-files (*.mat)'}, ...
            sprintf('Pick trial metrics file(s)  [%d selected]  -  Cancel to finish', ...
                    numel(files)), ...
            [d filesep], 'MultiSelect', 'on');
        if isequal(fn, 0)
            break
        end
        if ischar(fn), fn = {fn}; end
        files = [files, cellfun(@(x) fullfile(pth, x), fn, 'UniformOutput', false)];
        d = pth;
    end
    files = unique(files, 'stable');
    if ~isempty(files)
        fprintf('Picked %d trial file(s).\n', numel(files));
    end
end

% ----------------------------------------------------------------------
function tf = can_show_dialogs()
    tf = usejava('awt') && ~batchStartupOptionUsed;   % no picker under -batch
    try
        tf = tf && feature('ShowFigureWindows');
    catch
    end
end

% ======================================================================
function files = resolve_trial_files(trials, channel)
% Turn the 'trials' spec into a list of metrics .mat files.
    files = {};
    if isempty(trials)
        d = [dir(fullfile(pwd, 'grapes_response_metrics_*.mat')); ...
             dir(fullfile(pwd, '*', 'grapes_response_metrics_*.mat'))];
        files = fullfile({d.folder}, {d.name});
    else
        trials = cellstr(trials);
        for i = 1:numel(trials)
            t = trials{i};
            if exist(t, 'dir')
                if ~isempty(channel)
                    cand = fullfile(t, sprintf('grapes_response_metrics_ch%d.mat', channel));
                    if exist(cand, 'file')
                        files{end+1} = cand;
                        continue
                    end
                end
                dd = dir(fullfile(t, 'grapes_response_metrics_*.mat'));
                if isscalar(dd)
                    files{end+1} = fullfile(dd.folder, dd.name);
                elseif isempty(dd)
                    warning('no grapes_response_metrics_*.mat in %s', t);
                else
                    error(['%d metrics files in %s - pass ''channel'' or give ' ...
                           'explicit file paths'], numel(dd), t);
                end
            elseif exist(t, 'file')
                files{end+1} = t;
            else
                warning('trial not found: %s', t);
            end
        end
    end
    files = files(~cellfun(@isempty, files));
    % de-dup, keep order
    [~, ia] = unique(files, 'stable');
    files = files(ia);
end

% ======================================================================
function make_comparison_plots(cmp, labels, plot_dir, plot_metrics, show_images, pics_dir, max_plot_stims)
% PNG plots for a trial comparison. Always writes summary.png. For nstim <=
% max_plot_stims also writes change_<metric>.png + heatmap[_delta]_<metric>.png
% (+ per_stimulus/ if show_images); for more stimuli those are unreadable, so
% instead writes one scatter_<metric>.png per metric (reference trial vs each
% other trial, one point per stimulus).

    if nargin < 7 || isempty(max_plot_stims), max_plot_stims = 40; end
    if ~exist(plot_dir, 'dir'), mkdir(plot_dir); end
    labels  = cellstr(labels);
    ntrial  = numel(labels);
    stimN   = cmp.stim_number(:);
    stimNm  = cmp.stim_name(:);
    nstim   = numel(stimN);
    xt      = 1:ntrial;
    lab_short = shorten(labels);
    row_lab = arrayfun(@(k) sprintf('%d %s', stimN(k), stimNm(k)), (1:nstim)', ...
                       'UniformOutput', false);
    have_metric = @(m) isfield(cmp.wide, m) && any(isfinite(cmp.wide.(m)(:)));
    plot_metrics = plot_metrics(cellfun(have_metric, plot_metrics));
    col_key = strjoin(compose('%d=%s', xt(:), string(lab_short(:))), '   ');
    big = nstim > max_plot_stims;

    % ---- summary bar charts --------------------------------------------
    f = figure('Visible','off','Position',[100 100 max(1100, 300*ntrial) 720]);
    sm = cmp.summary;
    svars = {'n_responsive','mean_zscore','median_zscore','mean_auroc', ...
             'mean_resp_delta_fr_hz','frac_good_lat'};
    svars = svars(ismember(svars, sm.Properties.VariableNames));
    tl = tiledlayout(f, 'flow', 'Padding','loose','TileSpacing','loose');
    title(tl, sprintf('ch%d  %s  (flavour: %s)   %d stims', ...
          cmp.channel, cmp.class, cmp.grapes_tag, nstim), 'Interpreter','none');
    xlabel(tl, col_key, 'Interpreter','none');
    for k = 1:numel(svars)
        ax = nexttile(tl);
        bar(ax, xt, sm.(svars{k}));
        xlim(ax, [0.5 ntrial+0.5]); xticks(ax, xt);
        title(ax, svars{k}, 'Interpreter','none'); grid(ax,'on');
    end
    save_fig(f, fullfile(plot_dir, 'summary.png'));

    % ---- per-metric detail --------------------------------------------
    cmap = lines(max(nstim,7));
    ref  = [];
    if isfield(cmp,'reference') && ~isempty(cmp.reference), ref = cmp.reference; end

    for mi = 1:numel(plot_metrics)
        m = plot_metrics{mi};
        M = cmp.wide.(m);

        if big
            % --- scatter: reference (or col 1) vs every other column ---
            rc = ref; if isempty(rc), rc = 1; end
            others = setdiff(1:ntrial, rc);
            f = figure('Visible','off','Position',[100 100 max(900, 460*numel(others)) 520]);
            tl = tiledlayout(f, 1, numel(others), 'Padding','loose','TileSpacing','loose');
            title(tl, sprintf('ch%d %s - %s   (x = %s)', cmp.channel, cmp.class, m, ...
                  lab_short{rc}), 'Interpreter','none');
            for jj = 1:numel(others)
                c = others(jj);
                ax = nexttile(tl); hold(ax,'on');
                x = M(:,rc); y = M(:,c);
                good = isfinite(x) & isfinite(y);
                scatter(ax, x(good), y(good), 14, 'filled', 'MarkerFaceAlpha', 0.5);
                lo = min([x(good);y(good)]); hi = max([x(good);y(good)]);
                if isfinite(lo) && hi > lo
                    plot(ax, [lo hi], [lo hi], 'k--');
                    rl = robust_lims([x(good);y(good)]);
                    if ~isempty(rl), xlim(ax, rl); ylim(ax, rl); end
                end
                grid(ax,'on'); axis(ax,'square');
                xlabel(ax, lab_short{rc}, 'Interpreter','none');
                ylabel(ax, lab_short{c},  'Interpreter','none');
                title(ax, sprintf('%s  (n=%d)', lab_short{c}, nnz(good)), 'Interpreter','none');
            end
            save_fig(f, fullfile(plot_dir, sprintf('scatter_%s.png', m)));
            continue
        end

        % --- line: value vs trial, one line per stimulus (paired-slope) ---
        f = figure('Visible','off','Position',[100 100 max(1000, 260*ntrial) 640]);
        ax = axes('Parent', f); hold(ax,'on');
        for s = 1:nstim
            plot(ax, xt, M(s,:), '-o', 'Color', cmap(s,:), 'MarkerFaceColor', cmap(s,:));
        end
        hold(ax,'off'); grid(ax,'on');
        xlim(ax,[0.5 ntrial+0.5]); xticks(ax, xt); xticklabels(ax, lab_short);
        set(ax, 'TickLabelInterpreter', 'none');
        ylabel(ax, m, 'Interpreter','none');
        ttl = sprintf('ch%d %s - %s across columns  (line per stim)', cmp.channel, cmp.class, m);
        rl = robust_lims(M(:));
        if ~isempty(rl) && any(M(:) < rl(1) | M(:) > rl(2))
            ylim(ax, rl);
            ttl = sprintf('%s  [y clipped]', ttl);
        end
        title(ax, ttl, 'Interpreter','none');
        fit_axes(ax);
        save_fig(f, fullfile(plot_dir, sprintf('change_%s.png', m)));

        make_heatmap(M, row_lab, lab_short, ...
            sprintf('ch%d %s - %s', cmp.channel, cmp.class, m), false, ...
            fullfile(plot_dir, sprintf('heatmap_%s.png', m)));
        if isfield(cmp, 'delta') && isfield(cmp.delta, m)
            make_heatmap(cmp.delta.(m), row_lab, lab_short, ...
                sprintf('ch%d %s - %s  (delta vs %s)', cmp.channel, cmp.class, m, ...
                        lab_short{ref}), true, ...
                fullfile(plot_dir, sprintf('heatmap_delta_%s.png', m)));
        end
    end

    % ---- per-stimulus panels ----------------------------------------
    if show_images && ~big
        sdir = fullfile(plot_dir, 'per_stimulus');
        if ~exist(sdir,'dir'), mkdir(sdir); end
        panel_metrics = intersect({'zscore','auroc','resp_delta_fr_hz','p_value_sign', ...
                                   'peak_ifr','frac_active_trials'}, plot_metrics, 'stable');
        for s = 1:nstim
            f = figure('Visible','off','Position',[100 100 max(1500, 340*numel(panel_metrics)) 460]);
            tl = tiledlayout(f, 1, 1+numel(panel_metrics), ...
                             'Padding','loose','TileSpacing','loose');
            title(tl, sprintf('stim %d  %s   (ch%d %s)', stimN(s), stimNm(s), ...
                  cmp.channel, cmp.class), 'Interpreter','none');
            xlabel(tl, col_key, 'Interpreter','none');
            ax = nexttile(tl);
            img = try_read_image(pics_dir, stimNm(s));
            if isempty(img)
                text(ax, 0.5,0.5,{'(image not found)', char(stimNm(s))}, ...
                     'HorizontalAlignment','center','Interpreter','none');
                axis(ax,'off');
            else
                imshow(img, 'Parent', ax);
            end
            for k = 1:numel(panel_metrics)
                axk = nexttile(tl);
                bar(axk, xt, cmp.wide.(panel_metrics{k})(s,:)); grid(axk,'on');
                xlim(axk, [0.5 ntrial+0.5]); xticks(axk, xt);
                title(axk, panel_metrics{k}, 'Interpreter','none');
            end
            save_fig(f, fullfile(sdir, sprintf('stim_%d.png', stimN(s))));
        end
    end
end

% ----------------------------------------------------------------------
function fit_axes(ax)
% Shrink a single axes so its own decorations (title/labels/ticklabels) fit
% inside the figure - the reliable cure for print() clipping titles or
% rotated tick labels near the figure edges under software OpenGL.
    try
        drawnow;
        ti = ax.TightInset;
        ax.Position = [ti(1)+0.012, ti(2)+0.012, ...
                       1-ti(1)-ti(3)-0.024, 1-ti(2)-ti(4)-0.024];
    catch
    end
end

% ----------------------------------------------------------------------
function make_heatmap(M, row_lab, col_lab, ttl, diverging, out_png)
    nr    = size(M,1);
    rlw   = max(cellfun(@numel, cellstr(row_lab)));   % longest row label
    clw   = max(cellfun(@numel, cellstr(col_lab)));
    fs    = 8; if nr > 45, fs = 6; end
    fig_w = max(900, 150*numel(col_lab) + 9*clw + 7*rlw);
    fig_h = min(1500, max(320, 22*nr + 170));
    f = figure('Visible','off','Position',[100 100 fig_w fig_h]);
    ax = axes(f);
    imagesc(ax, M, 'AlphaData', ~isnan(M));
    colorbar(ax);
    v = M(isfinite(M));
    if diverging
        mx = quantile(abs(v), 0.98);
        if isempty(mx) || mx <= 0, mx = max(abs(v)); end
        if ~isempty(mx) && mx > 0, set_clim(ax, [-mx mx]); end
        colormap(ax, diverging_cmap());
    else
        rl = robust_lims(v);
        if ~isempty(rl), set_clim(ax, rl); end
        colormap(ax, parula);
    end
    xticks(ax, 1:numel(col_lab)); xticklabels(ax, col_lab);
    yticks(ax, 1:nr); yticklabels(ax, row_lab);
    set(ax, 'TickLabelInterpreter','none', 'FontSize', fs);
    title(ax, ttl, 'Interpreter','none');
    % explicit margins: left for row labels, right for colorbar, bottom for
    % col labels, top for title (fit_axes would push the colorbar off-frame)
    L = min(0.45, 0.03 + (7*rlw)/fig_w);
    B = min(0.30, 0.06 + (9*clw)/fig_h);
    ax.Position = [L, B, max(0.30, 0.86 - L - 0.12), max(0.30, 0.92 - B - 0.06)];
    save_fig(f, out_png);
end

% ----------------------------------------------------------------------
function set_clim(ax, lims)
    try
        clim(ax, lims);
    catch
        caxis(ax, lims);
    end
end

% ----------------------------------------------------------------------
function lims = robust_lims(v)
% View limits that ignore a few blow-up outliers (e.g. zscore -> 1e10 when
% baseline variance is 0). Returns [] when the data is (near-)constant.
    lims = [];
    v = v(isfinite(v));
    if numel(v) < 3, return; end
    q = quantile(v, [0.02 0.98]);
    if q(2) <= q(1)
        q = [min(v) max(v)];
        if q(2) <= q(1), return; end
    end
    pad  = 0.08 * (q(2) - q(1));
    lims = [q(1)-pad, q(2)+pad];
end

% ----------------------------------------------------------------------
function save_fig(f, out_png)
    set(f, 'Color', 'w');
    % print() captures the whole figure rectangle, so titles/rotated tick
    % labels near the edges are never tight-cropped away (which exportgraphics
    % can do, especially under software OpenGL).
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

% ----------------------------------------------------------------------
function img = try_read_image(pics_dir, name)
    img = [];
    if isempty(pics_dir) || strlength(name) == 0, return; end
    cand = fullfile(pics_dir, char(name));
    if ~exist(cand, 'file')
        [~, b] = fileparts(char(name));
        d = dir(fullfile(pics_dir, [b '.*']));
        if isempty(d), return; end
        cand = fullfile(d(1).folder, d(1).name);
    end
    try
        img = imread(cand);
    catch
        img = [];
    end
end

% ----------------------------------------------------------------------
function lab = label_from_meta(meta, file)
% Default trial label = first path component of meta.working_dir below
% meta.session_root. Falls back to the metrics file's parent folder name.
    lab = '';
    if isstruct(meta) && isfield(meta, 'session_root') && isfield(meta, 'working_dir')
        sr = regexprep(char(meta.session_root), '[/\\]+$', '');
        wd = regexprep(char(meta.working_dir),  '[/\\]+$', '');
        if strncmp(wd, sr, numel(sr))
            rel = wd(numel(sr)+1:end);
            parts = regexp(rel, '[/\\]', 'split');
            parts = parts(~cellfun(@isempty, parts) & ~strcmp(parts, '.'));
            if ~isempty(parts), lab = parts{1}; end
        end
    end
    if isempty(lab)
        [d, ~] = fileparts(file);
        [~, lab] = fileparts(d);
    end
end

% ----------------------------------------------------------------------
function s = shorten(c, varargin)
% Tidy labels for tick display: collapse whitespace to '_', keep them unique.
% No truncation - callers set 'TickLabelInterpreter','none' so '_' stays '_'
% and long names are not clipped.
    s = cellfun(@(x) regexprep(x, '\s+', '_'), cellstr(c), 'UniformOutput', false);
    s = matlab.lang.makeUniqueStrings(s);
end

% ----------------------------------------------------------------------
function m = diverging_cmap()
% Simple blue-white-red diverging map (no toolbox dependency).
    t = linspace(0, 1, 128)';
    blue_half  = [t, t, ones(128,1)];              % blue  -> white
    red_half   = [ones(128,1), flipud(t), flipud(t)]; % white -> red
    m = [blue_half; red_half(2:end,:)];
end

% ----------------------------------------------------------------------
function tags = struct_metric_tags(s)
% Field names of struct s whose value is itself a struct with a .metrics table.
    tags = {};
    fn = fieldnames(s);
    for i = 1:numel(fn)
        v = s.(fn{i});
        if isstruct(v) && isfield(v, 'metrics') && istable(v.metrics)
            tags{end+1} = fn{i};
        end
    end
end
