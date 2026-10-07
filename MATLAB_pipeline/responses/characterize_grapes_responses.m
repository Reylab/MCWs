function out = characterize_grapes_responses(varargin)
% CHARACTERIZE_GRAPES_RESPONSES  Dump raster-response metrics to a .mat for testing.
%
% Run this from a folder that contains one or more grapes*.mat raster
% structures (e.g. a lightweight per-trial test folder). The session root
% (folder with NSx.mat) is found automatically via resolve_session_root,
% and experiment_properties_online3.mat / pics_used are read from there -
% so they do NOT need to be copied into every test folder. If pwd itself
% has those files they take precedence. The output .mat is written to pwd.
%
% It reproduces the metric computation used by plot_grapes_as_online
% (create_responses_data_parallel -> zscore / p_value_sign / latency ...)
% for a chosen channel set and stimulus set, WITHOUT plotting and WITHOUT
% ranking, and adds a handful of extra raster descriptors. Everything is
% written to a single lightweight .mat so many runs can be compared.
%
% Name/value parameters (mirrors plot_grapes_as_online where it makes sense)
%   'channels2plot' : 'all' (default) or numeric vector of channel numbers.
%                     >2000 -> -2000, >1000 -> -1000 (same as plot_grapes).
%   'stim_list'     : 'all' (default, = all pics used in scr_config_cell) or
%                     numeric vector of stimulus numbers.
%   'grapes_files'  : 'all' (default, every grapes*.mat in pwd, or in the
%                     session root if pwd has none) or a cell array of names.
%                     This includes grapes*_rescue.mat when present (written
%                     by do_structure_rescue_BCM_online3): its tag becomes
%                     out.grapes_rescue / out.grapes_blanks_rescue and its
%                     metrics table carries both classN and classN_rescue.
%                     Use compare_grapes_response_trials('mode','rescue') to
%                     line the pre/post-rescue flavours up.
%   'session_root'  : '' (default, auto-detect) or an explicit path to the
%                     folder holding NSx.mat / experiment_properties / pics_used.
%   'unit_label'    : {'mu','class'} (default) - which raster fields to score.
%   'outfile'       : base name of the .mat to write; the channel spec is
%                     appended before the extension, e.g. default
%                     'grapes_response_metrics.mat' with channels2plot=368
%                     -> 'grapes_response_metrics_ch368.mat'.
%   'save_ifr'      : false (default). If true, also stores the per-row IFR
%                     trace (heavy - only for debugging a single folder).
%   'priority_chs_ranking' : [] (default) or vector, forwarded to
%                     create_responses_data_parallel.
%
% Output (also saved to 'outfile'):
%   out.meta                - run metadata (dir, date, requested channels/stims)
%   out.stim_set            - stimulus numbers actually scored
%   out.stim_names          - names for those stimuli
%   out.files               - cell list of grapes files processed
%   out.(grapes_tag).metrics    - table, one row per (channel,class,stim)
%   out.(grapes_tag).resp_conf  - response-window config used
%   out.(grapes_tag).rank_config- macros returned by create_responses_data_parallel
%
% Metrics table columns
%   identity : channel, channel_num, ch_label, class, stim_number, stim_name
%   ranking  : zscore, median_post, mu_base, std_base, p_value_sign,
%              min_spk_test, p_test, good_lat, onset, tons, dura, ntrials,
%              ifr_thr
%   extra    : mean_fr_base_hz, mean_fr_post_hz, resp_delta_fr_hz,
%              spk_post_mean, spk_post_std, frac_active_trials,
%              first_spk_lat_med, auroc, p_ranksum, peak_ifr, peak_ifr_lat

    ipr = inputParser;
    addParameter(ipr, 'channels2plot', 'all');
    addParameter(ipr, 'stim_list', 'all');
    addParameter(ipr, 'grapes_files', 'all');
    addParameter(ipr, 'unit_label', {'mu','class'});
    addParameter(ipr, 'outfile', 'grapes_response_metrics.mat');
    addParameter(ipr, 'save_ifr', false, @islogical);
    addParameter(ipr, 'priority_chs_ranking', []);
    addParameter(ipr, 'session_root', '');   % override; else auto (folder with NSx.mat)
    parse(ipr, varargin{:});
    p = ipr.Results;

    SCRIPT_VERSION = 'characterize_grapes_responses v1 (2026-08-30)';

    % ---- make sure pipeline functions are reachable -----------------------
    pipeline_root = fileparts(fileparts(mfilename('fullpath')));   % .../MATLAB_pipeline
    addpath(genpath(pipeline_root));

    % ---- channel selection / offset (as in plot_grapes_as_online) --------
    channels2plot = p.channels2plot;
    if ~(ischar(channels2plot) || isstring(channels2plot))
        channels2plot = double(channels2plot(:)');
        if min(channels2plot) > 2000
            channels2plot = channels2plot - 2000;
        elseif min(channels2plot) > 1000
            channels2plot = channels2plot - 1000;
        end
    end

    % ---- output file name: append the channel spec before .mat ----------
    [out_dir, out_base, out_ext] = fileparts(p.outfile);
    if isempty(out_ext), out_ext = '.mat'; end
    ch_tag = channel_tag(p.channels2plot);
    outfile_final = fullfile(out_dir, [out_base '_' ch_tag out_ext]);

    % ---- resolve the session root -------------------------------------
    % The test folder only needs the grapes*.mat files. experiment_properties
    % and pics_used are read from the session root (folder with NSx.mat),
    % exactly like plot_grapes_as_online -> resolve_session_root.
    if ~isempty(p.session_root)
        root = char(p.session_root);
        if ~exist(root, 'dir')
            error('characterize_grapes_responses:badRoot', ...
                  'session_root does not exist: %s', root);
        end
    else
        try
            root = resolve_session_root();
        catch
            root = pwd;   % last resort: everything is expected in pwd
        end
    end
    pics_dir = fullfile(root, 'pics_used');

    % ---- experiment properties (pwd first, then session root) ----------
    ep_local = fullfile(pwd,  'experiment_properties_online3.mat');
    ep_root  = fullfile(root, 'experiment_properties_online3.mat');
    if exist(ep_local, 'file')
        ep_path = ep_local;
    elseif exist(ep_root, 'file')
        ep_path = ep_root;
    else
        error('characterize_grapes_responses:noExpProps', ...
              ['experiment_properties_online3.mat not found in pwd (%s) ' ...
               'or session root (%s)'], pwd, root);
    end
    fprintf('experiment_properties: %s\n', ep_path);
    ep = load(ep_path);
    experiment      = ep.experiment;
    scr_config_cell = ep.scr_config_cell;
    scr_end_cell    = ep.scr_end_cell;
    if isfield(ep, 'priority_chs_ranking') && isempty(p.priority_chs_ranking)
        p.priority_chs_ranking = ep.priority_chs_ranking;
    end
    final_n_scr = numel(scr_end_cell);

    % ---- which grapes files (pwd first, then session root) -------------
    if ischar(p.grapes_files) || isstring(p.grapes_files)
        if strcmpi(p.grapes_files, 'all')
            d = dir(fullfile(pwd, 'grapes*.mat'));
            if isempty(d) && ~strcmp(root, pwd)
                d = dir(fullfile(root, 'grapes*.mat'));
            end
            grapes_files = fullfile({d.folder}, {d.name});
        else
            grapes_files = local_locate(cellstr(p.grapes_files), pwd, root);
        end
    else
        grapes_files = local_locate(cellstr(p.grapes_files), pwd, root);
    end
    grapes_files = grapes_files(~cellfun(@isempty, grapes_files));
    % never treat our own output files (any channel tag) as input raster files
    [~, gnames_only] = cellfun(@fileparts, grapes_files, 'UniformOutput', false);
    grapes_files = grapes_files(~startsWith(gnames_only, out_base));
    if isempty(grapes_files)
        error('characterize_grapes_responses:noGrapes', ...
              'No grapes*.mat files found in pwd (%s) or session root (%s)', pwd, root);
    end

    % ---- IFR calculator constants --------------------------------------
    SR = 30000;

    out = struct();
    out.meta = struct('script_version', SCRIPT_VERSION, ...
                      'created', char(datetime('now','Format','yyyy-MM-dd HH:mm:ss')), ...
                      'working_dir', pwd, ...
                      'session_root', root, ...
                      'pics_dir', pics_dir, ...
                      'experiment_properties', ep_path, ...
                      'channels_requested', p.channels2plot, ...
                      'channels_resolved', channels2plot, ...
                      'stim_list_requested', p.stim_list, ...
                      'unit_label', {cellstr(p.unit_label)}, ...
                      'final_n_scr', final_n_scr, ...
                      'outfile', outfile_final);
    out.files_found = grapes_files(:)';
    stim_set_saved = [];

    processed = {};
    for gi = 1:numel(grapes_files)
        gname = grapes_files{gi};
        [~, gbase, gext] = fileparts(gname);
        tag = matlab.lang.makeValidName(gbase);
        fprintf('\n=== %s%s ===\n', gbase, gext);
      try
        % blanks flavour drives the response windows, same values as plot_grapes.
        % match 'blank' as well as 'blanks' - a file called grapes_blank.mat was
        % silently scored with the plain (non-blanks) windows.
        is_blanks = contains(gbase, 'blank', 'IgnoreCase', true);
        resp_conf = default_resp_conf(is_blanks);
        if is_blanks && ~contains(gbase, 'blanks')
            warning(['characterize_grapes_responses: "%s" looks like a blanks raster but ' ...
                     'is not named *blanks* - scoring it with the blanks windows. Rename it ' ...
                     'grapes_blanks.mat to match the pipeline convention.'], gbase);
        end

        grapes = load(gname);
        if ~isfield(grapes, 'rasters')
            warning('%s has no .rasters field - skipped', gname);
            continue
        end
        time_pre = getfielddef(grapes, 'time_pre', 1e3);
        time_pos = getfielddef(grapes, 'time_pos', 2e3);

        % restrict channels
        if ~(ischar(channels2plot) || isstring(channels2plot))
            new_rasters = struct();
            for i = 1:numel(channels2plot)
                cf = ['chan' num2str(channels2plot(i))];
                if isfield(grapes.rasters, cf)
                    new_rasters.(cf) = grapes.rasters.(cf);
                else
                    warning('%s: %s not present in %s', gname, cf, gname);
                end
            end
            grapes.rasters = new_rasters;
        end
        if isempty(fieldnames(grapes.rasters))
            warning('%s: no requested channels present - skipped', gname);
            continue
        end

        % stimulus set
        if ischar(p.stim_list) || isstring(p.stim_list)
            if ~contains(grapes.exp_type, 'freq_tag')
                stim_set = unique(cell2mat(cellfun(@(x) x.pics2use, scr_config_cell, ...
                                                   'UniformOutput', false)));
            else
                stim_set = unique(scr_config_cell{1,1}.order_pic);
            end
        else
            stim_set = unique(double(p.stim_list(:)'));
        end
        stim_set = stim_set(:)';
        stim_set_saved = stim_set;

        ifr_calc = IFRCalculator(3.035, 10, 1, SR, time_pre, time_pos);

        [data, rank_config] = create_responses_data_parallel( ...
                                  grapes, stim_set, cellstr(p.unit_label), ...
                                  ifr_calc, resp_conf, [], p.priority_chs_ranking);

        if isempty(data)
            warning('%s: create_responses_data_parallel returned nothing', gname);
            continue
        end

        metrics = build_metrics_table(data, grapes, experiment, resp_conf, ...
                                      ifr_calc, p.save_ifr);

        out.(tag).metrics     = metrics;
        out.(tag).resp_conf   = resp_conf;
        out.(tag).rank_config = rank_config;
        out.(tag).is_blanks   = is_blanks;
        out.(tag).time_pre    = time_pre;
        out.(tag).time_pos    = time_pos;

        fprintf('  %d rows  (%d channels x %d units x %d stims)\n', ...
                height(metrics), numel(unique(metrics.channel)), ...
                numel(unique(metrics.class)), numel(stim_set));
        fprintf('  responsive (min_spk_test & p_test & good_lat): %d\n', ...
                sum(metrics.min_spk_test & metrics.p_test & metrics.good_lat));
        processed{end+1} = tag;
      catch ME
        warning('characterize_grapes_responses:fileFailed', ...
                'skipped %s: %s', gname, ME.message);
      end
    end
    out.files = processed(:)';
    if isempty(processed)
        error('characterize_grapes_responses:allFailed', ...
              'No grapes file could be processed in %s', pwd);
    end

    out.stim_set   = stim_set_saved;
    out.stim_names = stim_name_list(experiment, stim_set_saved);

    save(outfile_final, '-struct', 'out', '-v7');
    if isempty(fileparts(outfile_final))
        fprintf('\nSaved %s\n', fullfile(pwd, outfile_final));
    else
        fprintf('\nSaved %s\n', outfile_final);
    end
end

% ----------------------------------------------------------------------
function t = channel_tag(chan_req)
% Filename-safe tag for the requested channel(s): 'chAll' | 'ch368' | 'ch368-371'
    if ischar(chan_req) || isstring(chan_req)
        t = ['ch' matlab.lang.makeValidName(char(chan_req))];
        return
    end
    c = round(double(chan_req(:)'));
    if isscalar(c)
        t = sprintf('ch%d', c);
    else
        t = ['ch' strjoin(arrayfun(@(x) sprintf('%d', x), c, 'UniformOutput', false), '-')];
    end
end

% ======================================================================
function rc = default_resp_conf(is_blanks)
    rc = struct();
    if is_blanks
        rc.tmin_median = 50;   rc.tmax_median = 550;
        rc.tmin_base   = -550; rc.tmax_base   = -50;
    else
        rc.tmin_median = 200;  rc.tmax_median = 700;
        rc.tmin_base   = -900; rc.tmax_base   = -100;
    end
    rc.from_onset          = 1;
    rc.smooth_bin          = 1500;
    rc.min_spk_median      = 1;
    rc.psign_thr           = 0.05;
    rc.t_down              = 20;
    rc.over_threshold_time = 75;
    rc.below_threshold_time= 100;
    rc.nstd                = 3;
    rc.win_cent            = 1;
    rc.FR_resol            = 10;   % ms
    rc.min_ifr_thr         = 4;    % Hz
end

% ----------------------------------------------------------------------
function v = getfielddef(s, f, d)
    if isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end

% ----------------------------------------------------------------------
function paths = local_locate(names, dir1, dir2)
% Resolve each name to a full path: absolute as given, else in dir1, else dir2.
    paths = cell(size(names));
    for i = 1:numel(names)
        nm = char(names{i});
        [pp, ~] = fileparts(nm);
        if ~isempty(pp) && exist(nm, 'file')
            paths{i} = nm;
        elseif exist(fullfile(dir1, nm), 'file')
            paths{i} = fullfile(dir1, nm);
        elseif exist(fullfile(dir2, nm), 'file')
            paths{i} = fullfile(dir2, nm);
        else
            warning('grapes file not found: %s', nm);
            paths{i} = '';
        end
    end
end

% ----------------------------------------------------------------------
function names = stim_name_list(experiment, stim_set)
    names = strings(numel(stim_set), 1);
    if isempty(stim_set), return; end
    try
        raw = experiment.ImageNames.name(stim_set);
        names = string(raw);
        names = names(:);
    catch
        % leave empty strings
    end
end

% ======================================================================
function T = build_metrics_table(data, grapes, experiment, resp_conf, ifr_calc, save_ifr)
% Take the table returned by create_responses_data_parallel and enrich it
% with human-readable identity columns + extra raster descriptors computed
% straight from grapes.rasters.

    n = height(data);

    channel      = string(data.channel);
    class        = string(data.class);
    stim_number  = double(data.stim_number);
    channel_num  = zeros(n,1);
    ch_label     = strings(n,1);
    stim_name    = strings(n,1);

    mean_fr_base_hz   = nan(n,1);
    mean_fr_post_hz   = nan(n,1);
    resp_delta_fr_hz  = nan(n,1);
    spk_post_mean     = nan(n,1);
    spk_post_std      = nan(n,1);
    frac_active_trials= nan(n,1);
    first_spk_lat_med = nan(n,1);
    auroc             = nan(n,1);
    p_ranksum         = nan(n,1);
    peak_ifr          = nan(n,1);
    peak_ifr_lat      = nan(n,1);

    base_win_s = (resp_conf.tmax_base  - resp_conf.tmin_base ) / 1000;
    post_win_s = (resp_conf.tmax_median - resp_conf.tmin_median) / 1000;
    ejex = ifr_calc.ejex;
    ifr_peak_mask = ejex >= 0 & ejex <= (resp_conf.tmax_median + 300);

    try
        img_names = experiment.ImageNames.name;
    catch
        img_names = {};
    end

    for i = 1:n
        ch = char(channel(i));
        cl = char(class(i));
        sn = stim_number(i);

        m = regexp(ch, '\d+', 'match', 'once');
        if ~isempty(m), channel_num(i) = str2double(m); end
        try
            ch_label(i) = string(grapes.rasters.(ch).details.ch_label);
        catch
        end
        if ~isempty(img_names) && sn >= 1 && sn <= numel(img_names)
            try
                stim_name(i) = string(img_names(sn));
            catch
            end
        end

        try
            trials = grapes.rasters.(ch).(cl).stim{sn};
        catch
            continue
        end
        if isempty(trials), continue; end
        if ~iscell(trials), trials = num2cell(trials, 2); end

        nt = numel(trials);
        base_cnt = zeros(nt,1);
        post_cnt = zeros(nt,1);
        first_lat = nan(nt,1);
        for t = 1:nt
            s = trials{t};
            s = s(:).';
            s = s(s > -9000 & s < 9000);       % drop 9999 / 10000 sentinels
            bs = s(s >= resp_conf.tmin_base   & s <= resp_conf.tmax_base);
            ps = s(s >= resp_conf.tmin_median & s <= resp_conf.tmax_median);
            base_cnt(t) = numel(bs);
            post_cnt(t) = numel(ps);
            if ~isempty(ps), first_lat(t) = min(ps); end
        end

        mean_fr_base_hz(i)  = mean(base_cnt) / base_win_s;
        mean_fr_post_hz(i)  = mean(post_cnt) / post_win_s;
        resp_delta_fr_hz(i) = mean_fr_post_hz(i) - mean_fr_base_hz(i);
        spk_post_mean(i)    = mean(post_cnt);
        spk_post_std(i)     = std(post_cnt);
        frac_active_trials(i) = mean(post_cnt > 0);
        first_spk_lat_med(i)  = median(first_lat(~isnan(first_lat)));
        auroc(i) = auc_fast(base_cnt, post_cnt);
        p_ranksum(i) = safe_ranksum(base_cnt, post_cnt);

        try
            ifr = double(data.IFR{i});
            if numel(ifr) == numel(ejex)
                [pk, ix] = max(ifr(ifr_peak_mask));
                idxs = find(ifr_peak_mask);
                peak_ifr(i) = pk;
                peak_ifr_lat(i) = ejex(idxs(ix));
            end
        catch
        end
    end

    T = table(channel, channel_num, ch_label, class, stim_number, stim_name, ...
              double(data.zscore), double(data.median_post), ...
              double(data.mu_b), double(data.s_b), double(data.p_value_sign), ...
              logical(data.min_spk_test), logical(data.p_test), ...
              logical(data.good_lat), double(data.onset), double(data.tons), ...
              double(data.dura), double(data.ntrials), double(data.IFR_thr), ...
              mean_fr_base_hz, mean_fr_post_hz, resp_delta_fr_hz, ...
              spk_post_mean, spk_post_std, frac_active_trials, ...
              first_spk_lat_med, auroc, p_ranksum, peak_ifr, peak_ifr_lat, ...
        'VariableNames', {'channel','channel_num','ch_label','class', ...
              'stim_number','stim_name','zscore','median_post','mu_base', ...
              'std_base','p_value_sign','min_spk_test','p_test','good_lat', ...
              'onset','tons','dura','ntrials','ifr_thr','mean_fr_base_hz', ...
              'mean_fr_post_hz','resp_delta_fr_hz','spk_post_mean', ...
              'spk_post_std','frac_active_trials','first_spk_lat_med', ...
              'auroc','p_ranksum','peak_ifr','peak_ifr_lat'});

    if save_ifr && ismember('IFR', data.Properties.VariableNames)
        T.IFR = data.IFR;
        T.Properties.UserData.ifr_ejex = ejex;
    end
end

% ----------------------------------------------------------------------
function a = auc_fast(neg, pos)
% Area under the ROC curve for post (pos) vs baseline (neg) spike counts.
% 0.5 = no separation, 1 = post always > base.
    neg = neg(:); pos = pos(:);
    if isempty(neg) || isempty(pos), a = NaN; return; end
    try
        r = tiedrank([neg; pos]);
    catch
        [~, ord] = sort([neg; pos]); r = zeros(numel(ord),1); r(ord) = 1:numel(ord);
    end
    n1 = numel(pos);
    a = (sum(r(numel(neg)+1:end)) - n1*(n1+1)/2) / (numel(neg)*n1);
end

% ----------------------------------------------------------------------
function pv = safe_ranksum(base_cnt, post_cnt)
    pv = NaN;
    if numel(base_cnt) < 2 || numel(post_cnt) < 2, return; end
    try
        pv = ranksum(base_cnt, post_cnt, 'tail', 'left');   % base < post
    catch
        try
            pv = ranksum(base_cnt, post_cnt);
        catch
        end
    end
end
