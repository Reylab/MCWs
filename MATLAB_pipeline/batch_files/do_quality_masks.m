function do_quality_masks(channels, varargin)
% DO_QUALITY_MASKS - Combined pre-processing quality control pass.
%
% Orchestrates the three spike-quality masking steps:
%   - compute_bundle_artifact_mask  (cross-channel collision -> mask_nonart)
%   - compute_within_channel_mask   (waveform QC -> mask_non_quarantine)
%   - compute_refract_viol_mask     (refractory chains -> mask_non_refract)
%
% Each step is its own function below (to be split into separate files
% later). This parent loads each channel's *_spikes.mat ONCE, calls whichever
% steps are enabled, combines the resulting masks, recomputes index/spikes,
% and saves once per channel.
%
% Intended to be run BEFORE Do_features / Do_clustering.
%
% Inputs:
%   channels - vector of channel IDs (NSx.chan_ID values)
%
% Optional parameters (addParameter):
%   'do_bundle'    , true/false            (default: true)
%   'do_within'    , true/false            (default: true)
%   'do_refract'   , true/false            (default: true)
%   't_win'        , scalar                (default: 0.5)   - passed to compute_bundle_artifact_mask
%   'bundle_min_art', scalar               (default: 6)     - passed to compute_bundle_artifact_mask
%   'qc_params'    , struct                (default: see compute_within_channel_mask) - passed through
%   'keep_strategy', 'none'/'first'/'last' (default: 'none') - passed to compute_refract_viol_mask
%
% Usage:
%   do_quality_masks(1:16)
%   do_quality_masks(1:16, 'do_refract', false)
%   do_quality_masks(1:16, 'keep_strategy', 'first')

do_quality_masks_tic = tic;

% ---- Parse optional arguments ----
p = inputParser;
addParameter(p, 'do_bundle',  true, @islogical);
addParameter(p, 'do_within',  true, @islogical);
addParameter(p, 'do_refract', true, @islogical);
addParameter(p, 't_win', 0.5, @isnumeric);
addParameter(p, 'bundle_min_art', 6, @isnumeric);
addParameter(p, 'qc_params', struct(), @isstruct);
addParameter(p, 'keep_strategy', 'none', @ischar);
parse(p, varargin{:});

do_bundle      = p.Results.do_bundle;
do_within      = p.Results.do_within;
do_refract     = p.Results.do_refract;
t_win          = p.Results.t_win;
bundle_min_art = p.Results.bundle_min_art;
qc_params_in   = p.Results.qc_params;
keep_strategy  = p.Results.keep_strategy;

% ---- Resolve channels / spikes folder ----
load('NSx','NSx');
NSx_proc = NSx(ismember(cell2mat({NSx.chan_ID}), channels));

dates = dir(fullfile(pwd, 'spikes*'));
dates = dates([dates.isdir]);
if isempty(dates), error('No spikes folders found.'); end
[~, idx] = max([dates.datenum]);
active_spikes_dir = fullfile(pwd, dates(idx).name);

num_channels_proc = length(NSx_proc);

% ---- Single load per channel ----
chan_data = cell(num_channels_proc, 1);
chan_file = cell(num_channels_proc, 1);
for k = 1:num_channels_proc
    ch_lbl = NSx_proc(k).output_name;
    chan_file{k} = fullfile(active_spikes_dir, sprintf('%s_spikes.mat', ch_lbl));
    chan_data{k} = load(chan_file{k});
end

% =====================================================================
% STEP 1: Bundle / cross-channel artifact detection
% =====================================================================
if do_bundle
    chan_data = compute_bundle_artifact_mask(NSx_proc, chan_data, ...
        't_win', t_win, 'bundle_min_art', bundle_min_art);
end

% =====================================================================
% STEP 2: Within-channel waveform QC
% =====================================================================
if do_within
    for k = 1:num_channels_proc
        ch_lbl = NSx_proc(k).output_name;
        SPK = chan_data{k};
        spikes_all = get_spikes_all(SPK);

        if isempty(fieldnames(qc_params_in))
            [mask_non_quarantine, quarantine_properties, qc_params_used] = ...
                compute_within_channel_mask(spikes_all);
        else
            [mask_non_quarantine, quarantine_properties, qc_params_used] = ...
                compute_within_channel_mask(spikes_all, 'qc_params', qc_params_in);
        end

        SPK.mask_non_quarantine = mask_non_quarantine;
        SPK.quarantine_properties = quarantine_properties;
        SPK.par.qc_params = qc_params_used;
        chan_data{k} = SPK;

        fprintf('Within-channel QC: %s - quarantined %d/%d spikes\n', ...
            ch_lbl, sum(~mask_non_quarantine), numel(mask_non_quarantine));
    end
end

% =====================================================================
% STEP 3: Refractory violation
% =====================================================================
if do_refract
    for k = 1:num_channels_proc
        ch_lbl = NSx_proc(k).output_name;
        SPK = chan_data{k};
        index_all = get_index_all(SPK);

        mask_non_refract = compute_refract_viol_mask(index_all, SPK.par, ...
            'keep_strategy', keep_strategy);

        SPK.mask_non_refract = mask_non_refract;
        chan_data{k} = SPK;

        fprintf('Refractory check: %s - %d/%d spikes flagged (%.2f%%)\n', ...
            ch_lbl, sum(~mask_non_refract), numel(mask_non_refract), ...
            sum(~mask_non_refract)/numel(mask_non_refract)*100);
    end
end

% =====================================================================
% Combine masks, recompute index/spikes, single save per channel
% =====================================================================
for k = 1:num_channels_proc
    ch_lbl = NSx_proc(k).output_name;
    SPK = chan_data{k};

    index_all  = get_index_all(SPK);
    spikes_all = get_spikes_all(SPK);
    n = numel(index_all);

    mask_total_pass = true(1, n);
    if isfield(SPK, 'mask_nonart')
        mask_total_pass = mask_total_pass & logical(reshape(SPK.mask_nonart, 1, []));
    end
    if isfield(SPK, 'mask_non_quarantine')
        mask_total_pass = mask_total_pass & logical(reshape(SPK.mask_non_quarantine, 1, []));
    end
    if isfield(SPK, 'mask_non_refract')
        mask_total_pass = mask_total_pass & logical(reshape(SPK.mask_non_refract, 1, []));
    end
    if isfield(SPK, 'mask_taskspks')
        mask_total_pass = mask_total_pass & logical(reshape(SPK.mask_taskspks, 1, []));
    end

    SPK.index      = reshape(index_all(mask_total_pass), 1, []);
    SPK.spikes     = spikes_all(mask_total_pass, :);
    SPK.index_all  = index_all;
    SPK.spikes_all = spikes_all;

    chan_data{k} = SPK;

    fprintf('Channel %s: combined pass %d/%d (%.2f%%)\n', ...
        ch_lbl, sum(mask_total_pass), n, sum(mask_total_pass)/n*100);
end

for k = 1:num_channels_proc
    SPK = chan_data{k};
    fields_to_save = {'index','spikes','index_all','spikes_all','par'};
    if isfield(SPK,'mask_nonart'),           fields_to_save{end+1} = 'mask_nonart'; end
    if isfield(SPK,'mask_non_quarantine'),   fields_to_save{end+1} = 'mask_non_quarantine'; end
    if isfield(SPK,'quarantine_properties'), fields_to_save{end+1} = 'quarantine_properties'; end
    if isfield(SPK,'mask_non_refract'),      fields_to_save{end+1} = 'mask_non_refract'; end

    save_struct = struct();
    for f = 1:length(fields_to_save)
        save_struct.(fields_to_save{f}) = SPK.(fields_to_save{f});
    end

    save(chan_file{k}, '-struct', 'save_struct', '-append');
end

do_quality_masks_toc = toc(do_quality_masks_tic);
fprintf('do_quality_masks DONE in %.2f seconds.\n', do_quality_masks_toc);

end


% =========================================================================
% Small shared helpers for index_all/spikes_all fallback
% =========================================================================
function spikes_all = get_spikes_all(SPK)
    if isfield(SPK,'spikes_all')
        spikes_all = SPK.spikes_all;
    else
        spikes_all = SPK.spikes;
    end
end

function index_all = get_index_all(SPK)
    if isfield(SPK,'index_all')
        index_all = reshape(SPK.index_all, 1, []);
    else
        index_all = reshape(SPK.index, 1, []);
    end
end


% =========================================================================
% STEP 1 FUNCTION: compute_bundle_artifact_mask
% =========================================================================
function chan_data = compute_bundle_artifact_mask(NSx_proc, chan_data, varargin)
% COMPUTE_BUNDLE_ARTIFACT_MASK - Cross-channel collision artifact detection.
%
% For each bundle represented in NSx_proc, finds spikes co-occurring across
% >= bundle_min_art channels within t_win and flags them as artifacts.
% Writes mask_nonart (TRUE = not an artifact) into chan_data{k} for each
% channel, in-memory.
%
% Inputs:
%   NSx_proc  - NSx struct array, filtered to the channels of interest
%   chan_data - cell array of loaded spikes-file structs, one per NSx_proc entry
%
% Optional parameters:
%   't_win'          , scalar (default: 0.5)
%   'bundle_min_art' , scalar (default: 6)
%
% Output:
%   chan_data - same cell array, with mask_nonart (and par.t_win /
%               par.bundle_min_art) added to each struct
%
% NOTE: only bundle-mates present in NSx_proc are considered. If NSx_proc is
% a subset of a bundle, cross-channel detection will be incomplete for that
% bundle (same limitation as the original bundle_artifact.m).

p = inputParser;
addParameter(p, 't_win', 0.5, @isnumeric);
addParameter(p, 'bundle_min_art', 6, @isnumeric);
parse(p, varargin{:});

t_win = p.Results.t_win;
bundle_min_art = p.Results.bundle_min_art;

bundle_tic = tic;
bundles_to_explore = unique({NSx_proc.bundle});

for ibun = 1:length(bundles_to_explore)
    pos_chans_probe = find(arrayfun(@(x) strcmp(x.bundle, bundles_to_explore{ibun}), NSx_proc));
    posch = pos_chans_probe(1);
    if ~NSx_proc(posch).is_micro
        continue
    end

    all_spktimes = [];
    which_chan = [];
    for k = 1:length(pos_chans_probe)
        kk = pos_chans_probe(k);
        SPK = chan_data{kk};
        if isfield(SPK,'index_all')
            spktimes = reshape(SPK.index_all, 1, []);
        else
            spktimes = reshape(SPK.index, 1, []);
        end
        all_spktimes = [all_spktimes spktimes];
        which_chan = [which_chan NSx_proc(kk).chan_ID*ones(size(spktimes))];
    end
    [all_spktimes, II] = sort(all_spktimes);
    which_chan = which_chan(II);
    is_artifact = false(size(all_spktimes));
    artifact_idxs = [];

    b_parallel = true;
    num_cores = feature('numCores');
    if num_cores * 100 < numel(all_spktimes) && b_parallel
        split_size = floor(numel(all_spktimes) / num_cores);
        split_idx = 1:split_size:numel(all_spktimes);
        split_idx(end) = numel(all_spktimes);
        num_cores_actual = numel(split_idx) - 1;
        f_det(1:num_cores_actual) = parallel.FevalFuture;

        for i = 1:num_cores_actual
            start_idx = split_idx(i);
            end_idx = split_idx(i+1);

            buffer_spks = 0;
            if end_idx < numel(all_spktimes)
                while end_idx + buffer_spks < numel(all_spktimes) && ...
                      all_spktimes(end_idx + buffer_spks + 1) <= all_spktimes(end_idx) + t_win
                    buffer_spks = buffer_spks + 1;
                end
            end

            spktimes_chunk = all_spktimes(start_idx:end_idx + buffer_spks);
            whichchan_chunk = which_chan(start_idx:end_idx + buffer_spks);
            f_det(i) = parfeval(@local_detect_artifacts, 1, start_idx, split_size, ...
                                spktimes_chunk, whichchan_chunk, t_win, bundle_min_art);
        end

        for i = 1:num_cores_actual
            [~, art_idxs] = fetchNext(f_det);
            artifact_idxs = [artifact_idxs art_idxs];
        end
        clear f_det
    else
        for ispk = 1:numel(all_spktimes)
            which_spks = find(all_spktimes >= all_spktimes(ispk) & all_spktimes < all_spktimes(ispk) + t_win);
            if numel(unique(which_chan(which_spks))) >= bundle_min_art
                artifact_idxs = [artifact_idxs which_spks];
            end
        end
    end

    artifact_idxs = unique(artifact_idxs);
    is_artifact(artifact_idxs) = true;

    for k = 1:length(pos_chans_probe)
        kk = pos_chans_probe(k);
        ch_id = NSx_proc(kk).chan_ID;

        SPK = chan_data{kk};
        if isfield(SPK,'index_all')
            index_all_k = reshape(SPK.index_all, 1, []);
        else
            index_all_k = reshape(SPK.index, 1, []);
        end

        index_pass = all_spktimes(~is_artifact & (which_chan == ch_id));
        mask_nonart = ismember(index_all_k, index_pass);

        SPK.mask_nonart = reshape(mask_nonart, 1, []);
        SPK.par.t_win = t_win;
        SPK.par.bundle_min_art = bundle_min_art;
        chan_data{kk} = SPK;

        fprintf('Bundle artifact: %d/%d artifact spikes in %s\n', ...
            sum(~mask_nonart), numel(mask_nonart), NSx_proc(kk).output_name);
    end

    bundle_toc_bun = toc(bundle_tic);
    fprintf('Bundle %s: collision detection - total artifacts %d/%d (%.2f%%)\n', ...
        NSx_proc(posch).bundle, numel(artifact_idxs), numel(all_spktimes), ...
        numel(artifact_idxs)/numel(all_spktimes)*100);
end

bundle_toc = toc(bundle_tic);
fprintf('Bundle artifact step DONE in %.2f seconds.\n', bundle_toc);

end

function artifact_idxs = local_detect_artifacts(split_idx, splitsize, spktimes, ...
                                            whichchan, t_win, bundle_min_art)
    artifact_idxs = [];
    for ispk = 1:splitsize
        which_spks = find(spktimes >= spktimes(ispk) & spktimes < spktimes(ispk)+t_win);
        if numel(unique(whichchan(which_spks))) >= bundle_min_art
            which_spks = which_spks + split_idx - 1;
            artifact_idxs = [artifact_idxs which_spks];
        end
    end
end


% =========================================================================
% STEP 2 FUNCTION: compute_within_channel_mask
% =========================================================================
function [mask_non_quarantine, quarantine_properties, qc_params] = compute_within_channel_mask(spikes_all, varargin)
% COMPUTE_WITHIN_CHANNEL_MASK - Waveform shape/amplitude QC for one channel.
%
% Inputs:
%   spikes_all - [Nspikes x Nsamples] waveform matrix
%
% Optional parameters:
%   'qc_params', struct - any fields override the defaults below:
%       min_amplitude_percentile (default 5)
%       min_width_idx            (default 3)
%       max_width_idx            (default 15)
%       prominence_ratio_threshold  (default 0.01)
%       final_prominence_ratio_pass (default 2)
%
% Outputs:
%   mask_non_quarantine    - 1xN logical, TRUE = passes waveform QC
%   quarantine_properties  - struct of per-spike diagnostic properties
%   qc_params              - the qc_params actually used (defaults merged with overrides)

p = inputParser;
addParameter(p, 'qc_params', struct(), @isstruct);
parse(p, varargin{:});
qc_params_override = p.Results.qc_params;

qc_params = struct();
qc_params.min_amplitude_percentile = 5;
qc_params.min_width_idx = 3;
qc_params.max_width_idx = 15;
qc_params.prominence_ratio_threshold = 0.01;
qc_params.final_prominence_ratio_pass = 2;

override_fields = fieldnames(qc_params_override);
for f = 1:length(override_fields)
    qc_params.(override_fields{f}) = qc_params_override.(override_fields{f});
end

[mask_quarantine_local, quarantine_properties] = analyze_spike_waveforms(spikes_all, qc_params);
mask_non_quarantine = ~logical(reshape(mask_quarantine_local, 1, []));

end

function [quarantine_mask, quarantine_properties] = analyze_spike_waveforms(spikes, par)
    % Analyzes spikes using decision tree: amplitude -> peak count -> prominence -> width.
    % Copied verbatim from within_channel.m

    num_spikes = size(spikes, 1);
    sample_20_idx = 20;

    quarantine_mask = false(num_spikes, 1);

    prominence_ratio = nan(num_spikes, 1);
    num_peaks_arr = zeros(num_spikes, 1);
    prominence_sample_20 = nan(num_spikes, 1);
    other_prominence = nan(num_spikes, 1);
    width = nan(num_spikes, 1);
    peak_sample_20 = nan(num_spikes, 1);
    other_peak_loc = nan(num_spikes, 1);
    peak_pos_max = nan(num_spikes, 1);
    prominence_pos_max = nan(num_spikes, 1);
    low_low_amp_spike = false(num_spikes, 1);

    peak_info = cell(num_spikes, 1);

    if size(spikes, 2) < sample_20_idx
        warning('ArtifactRemoval:ShortWaveform', ...
            'Waveform length (%d) is shorter than sample_20_idx (%d). Quarantining all spikes.', ...
            size(spikes, 2), sample_20_idx);
        quarantine_mask(:) = true;
        quarantine_properties = struct();
        quarantine_properties.prominence_ratio = prominence_ratio(:);
        quarantine_properties.num_peaks = num_peaks_arr(:);
        quarantine_properties.prominence_sample_20 = prominence_sample_20(:);
        quarantine_properties.other_prominence = other_prominence(:);
        quarantine_properties.width = width(:);
        quarantine_properties.peak_sample_20 = peak_sample_20(:);
        quarantine_properties.other_peak_loc = other_peak_loc(:);
        quarantine_properties.peak_pos_max = peak_pos_max(:);
        quarantine_properties.prominence_pos_max = prominence_pos_max(:);
        quarantine_properties.low_low_amp_spike = low_low_amp_spike(:);
        return;
    end

    for i = 1:num_spikes
        waveform = spikes(i,:);

        [pos_peaks, pos_locs, ~, pos_prominences] = findpeaks(waveform);
        [neg_peaks_inv, neg_locs, ~, neg_prominences] = findpeaks(-waveform);
        neg_peaks = -neg_peaks_inv;

        pks  = [pos_peaks, neg_peaks];
        locs = [pos_locs, neg_locs];
        p    = [pos_prominences, neg_prominences];
        amps = [pos_peaks, abs(neg_peaks)];

        num_peaks_arr(i) = length(pks);

        if isempty(pks)
            quarantine_mask(i) = true;
            continue;
        end

        [main_pk_amp, main_idx] = max(amps);
        main_pk_loc = locs(main_idx);
        main_pk_prominence = p(main_idx);
        is_main_feature = (locs == main_pk_loc) & (pks == pks(main_idx));

        peak_sample_20(i) = waveform(min(sample_20_idx, numel(waveform)));

        main_pk_width = calc_baseline_width(waveform, main_pk_loc);
        width(i) = main_pk_width;
        prominence_sample_20(i) = main_pk_prominence;

        if ~isempty(pos_peaks)
            [peak_pos_max(i), max_pos_amp_idx] = max(pos_peaks);
            prominence_pos_max(i) = pos_prominences(max_pos_amp_idx);
        end

        if length(pks) == 1
            % Single peak: proceed to width check
        elseif length(pks) > 1
            other_prominences_arr = p(~is_main_feature);
            other_peaks_arr = locs(~is_main_feature);

            if ~isempty(other_prominences_arr)
                [max_other_prominence, max_prom_idx_local] = max(other_prominences_arr);
                other_prominence(i) = max_other_prominence;
                other_peak_loc(i) = other_peaks_arr(max_prom_idx_local);

                if max_other_prominence >= (par.prominence_ratio_threshold * main_pk_amp)
                    if max_other_prominence > 0
                        prominence_ratio(i) = main_pk_prominence / max_other_prominence;
                    else
                        prominence_ratio(i) = inf;
                    end

                    if isnan(prominence_ratio(i)) || prominence_ratio(i) > par.final_prominence_ratio_pass
                        % Ratio is good (or NaN exception), continue to width check
                    else
                        quarantine_mask(i) = true;
                    end
                end
            end
        end

        if isnan(width(i)) || width(i) < par.min_width_idx || width(i) > par.max_width_idx
            quarantine_mask(i) = true;
        end
    end

    low_low_amp_spike(~isfinite(low_low_amp_spike)) = false;

    abs_peak_amps = abs(peak_sample_20);
    valid_peak_amps = abs_peak_amps(isfinite(abs_peak_amps));
    if ~isempty(valid_peak_amps)
        low_amp_threshold = prctile(valid_peak_amps, par.min_amplitude_percentile);
        low_low_amp_spike = abs(peak_sample_20) < low_amp_threshold;
        low_low_amp_spike(~isfinite(low_low_amp_spike)) = false;
    end

    quarantine_properties = struct();
    quarantine_properties.prominence_ratio = prominence_ratio(:);
    quarantine_properties.num_peaks = num_peaks_arr(:);
    quarantine_properties.prominence_sample_20 = prominence_sample_20(:);
    quarantine_properties.other_prominence = other_prominence(:);
    quarantine_properties.width = width(:);
    quarantine_properties.peak_sample_20 = peak_sample_20(:);
    quarantine_properties.other_peak_loc = other_peak_loc(:);
    quarantine_properties.peak_pos_max = peak_pos_max(:);
    quarantine_properties.prominence_pos_max = prominence_pos_max(:);
    quarantine_properties.low_low_amp_spike = low_low_amp_spike(:);
end

function width_val = calc_baseline_width(waveform, peak_idx)
    % Measure width using a baseline-to-peak half-height rule.
    % Copied verbatim from within_channel.m
    width_val = nan;

    if numel(waveform) < peak_idx
        return;
    end

    n = numel(waveform);
    peak_voltage = waveform(peak_idx);

    h = min(5, n);
    first_seg = waveform(1:h);
    last_seg = waveform(max(1, n-h+1):n);
    baseline = mean([first_seg, last_seg]);

    half_amplitude = baseline + (peak_voltage - baseline) / 2;

    peak_idx_0based = peak_idx - 1;

    left_idx = nan;
    left_candidates = find(waveform(1:peak_idx) > half_amplitude);
    if ~isempty(left_candidates)
        ci_1based = left_candidates(end);
        ci_0based = ci_1based - 1;
        if (ci_0based < peak_idx_0based) && (ci_1based + 1 <= peak_idx)
            y0 = waveform(ci_1based + 1);
            y1 = waveform(ci_1based);
            x0 = ci_0based + 1;
            x1 = ci_0based;
            left_idx = interp1([y0, y1], [x0, x1], half_amplitude, 'linear', 'extrap');
        end
    end

    right_idx = nan;
    seg = waveform(peak_idx:end);
    right_candidates = find(seg > half_amplitude);
    if ~isempty(right_candidates)
        ci_local_1based = right_candidates(1);
        ci_local_0based = ci_local_1based - 1;
        ci_global_0based = peak_idx_0based + ci_local_0based;
        ci_global_1based = ci_global_0based + 1;
        if (ci_global_0based - 1 >= 0) && (ci_global_1based <= n)
            y_vals = waveform(ci_global_1based - 1 : ci_global_1based);
            x_vals = [ci_global_0based - 1, ci_global_0based];
            right_idx = interp1(y_vals, x_vals, half_amplitude, 'linear', 'extrap');
        end
    end

    if ~isnan(left_idx) && ~isnan(right_idx)
        width_val = right_idx - left_idx;
    end
end


% =========================================================================
% STEP 3 FUNCTION: compute_refract_viol_mask
% =========================================================================
function mask_non_refract = compute_refract_viol_mask(index_all, par, varargin)
% COMPUTE_REFRACT_VIOL_MASK - Refractory-period violation chain flagging.
%
% Inputs:
%   index_all - 1xN (or Nx1) vector of spike times (ms)
%   par       - parameters struct; must contain par.ref_ms (or par.ref + par.sr
%               as a fallback for computing ref_val, see notes below)
%
% Optional parameters:
%   'keep_strategy', 'none'/'first'/'last' (default: 'none')
%       'first' - keep chain start, flag the rest
%       'last'  - keep chain end, flag the rest
%       'none'  - flag every spike in the chain (default, conservative)
%
% Output:
%   mask_non_refract - 1xN logical, TRUE = not part of a refractory-violation chain
%
% NOTE: as in the original refract_viol.m, ref_val is taken directly from
% par.ref_ms (this overrides any par.ref/par.sr fallback computed above it,
% matching original behavior - par.ref_ms must be present).

p = inputParser;
addParameter(p, 'keep_strategy', 'none', @ischar);
parse(p, varargin{:});
keep_strategy = p.Results.keep_strategy;

index_all = reshape(index_all, [], 1); % column vector for diff/chain logic
n = numel(index_all);

if isfield(par, 'ref_ms')
    ref_val = par.ref_ms;
elseif isfield(par, 'ref')
    ref_val = par.ref / (par.sr / 1000);
else
    ref_val = 1.5;
end

% Matches original: par.ref_ms takes precedence if present (required)
ref_val = par.ref_ms;

if n > 1
    gaps = diff(index_all);
    in_chain_gap = gaps < ref_val;
    ext_gap = [false; in_chain_gap; false];
    chain_starts = find(diff(ext_gap) == 1);
    chain_ends   = find(diff(ext_gap) == -1);

    mask_refract = false(n, 1);

    switch lower(keep_strategy)
        case 'first'
            for c = 1:length(chain_starts)
                mask_refract(chain_starts(c) + 1 : chain_ends(c)) = true;
            end
        case 'last'
            for c = 1:length(chain_starts)
                mask_refract(chain_starts(c) : chain_ends(c) - 1) = true;
            end
        otherwise
            for c = 1:length(chain_starts)
                mask_refract(chain_starts(c) : chain_ends(c)) = true;
            end
    end
else
    mask_refract = false(n, 1);
end

mask_non_refract = reshape(~mask_refract, 1, []);

end