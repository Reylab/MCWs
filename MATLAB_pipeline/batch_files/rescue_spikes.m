function rescue_spikes(channels, varargin)
% rescue_spikes - Attempts to reclassify quarantined spikes using template matching
% against the already-clustered ("good") spike population.
%
% Inputs:
%   channels - vector of channel IDs (NSx.chan_ID values)
%   Optional:
%     'parallel', true/false        (default: false)
%     'restore', true/false         (default: false)
%     'peak_weight', scalar > 0     (default: 1)
%     'amp_dir', 'pos'/'neg'/'both'  (default: 'neg')
%     'quarantine_masks', cellstr   (default: {} -> auto-detect)
%         Any subset of: {'mask_nonart','mask_non_quarantine','mask_non_refract'}
%         A spike is treated as "quarantined" (a rescue candidate) if it is
%         FALSE in ANY of the selected/detected masks. mask_taskspks is never
%         used here.
%
% Usage:
%   rescue_spikes(channels, 'parallel', true)
%   rescue_spikes(channels, 'quarantine_masks', {'mask_non_refract'})
%   rescue_spikes(channels, 'restore', true)

% ---- Parse optional arguments ----
p = inputParser;
addParameter(p, 'parallel', false, @islogical);
addParameter(p, 'restore', false, @islogical);
addParameter(p, 'peak_weight', 1, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'amp_dir', 'neg', @ischar);
addParameter(p, 'quarantine_masks', {}, @iscell);
parse(p, varargin{:});

parallel          = p.Results.parallel;
restore           = p.Results.restore;
peak_weight       = p.Results.peak_weight;
amp_dir           = p.Results.amp_dir;
quarantine_masks  = p.Results.quarantine_masks;

% Known quarantine mask names (mask_taskspks deliberately excluded - handled elsewhere)
valid_quarantine_masks = {'mask_nonart','mask_non_quarantine','mask_non_refract'};
if ~isempty(quarantine_masks)
    unknown = setdiff(quarantine_masks, valid_quarantine_masks);
    if ~isempty(unknown)
        error('rescue_spikes: unrecognized quarantine_masks entries: %s', strjoin(unknown, ', '));
    end
end

% ---- Locate active spikes / times folders ----
dates_spikes = dir(fullfile(pwd, 'spikes*'));
dates_spikes = dates_spikes([dates_spikes.isdir]);
if isempty(dates_spikes), error('No spikes folders found.'); end
[~, idx_s] = max([dates_spikes.datenum]);
active_spikes_dir = fullfile(pwd, dates_spikes(idx_s).name);

dates_times = dir(fullfile(pwd, 'times*'));
dates_times = dates_times([dates_times.isdir]);
if isempty(dates_times)
    active_times_dir = active_spikes_dir;
    fprintf('No times folder found. Using spikes folder for times files: %s\n', active_spikes_dir);
else
    [~, idx_t] = max([dates_times.datenum]);
    active_times_dir = fullfile(pwd, dates_times(idx_t).name);
end

% ---- Resolve channels via NSx ----
load('NSx','NSx');
NSx_proc = NSx(ismember(cell2mat({NSx.chan_ID}), channels));
num_channels_proc = length(NSx_proc);

if restore
    fprintf('Starting rescue_spikes RESTORE on %d channels...\n', num_channels_proc);
else
    fprintf('Starting rescue_spikes on %d channels...\n', num_channels_proc);
end

if parallel
    parfor kk = 1:num_channels_proc
        process_channel_rescue(NSx_proc(kk), active_spikes_dir, active_times_dir, ...
            restore, peak_weight, amp_dir, quarantine_masks, valid_quarantine_masks);
    end
else
    for kk = 1:num_channels_proc
        process_channel_rescue(NSx_proc(kk), active_spikes_dir, active_times_dir, ...
            restore, peak_weight, amp_dir, quarantine_masks, valid_quarantine_masks);
    end
end

fprintf('rescue_spikes DONE.\n');
end

function process_channel_rescue(ch_info, active_spikes_dir, active_times_dir, ...
        restore, peak_weight, amp_dir, quarantine_masks, valid_quarantine_masks)

    ch_lbl = ch_info.output_name;
    spike_file = fullfile(active_spikes_dir, sprintf('%s_spikes.mat', ch_lbl));
    times_file = fullfile(active_times_dir, sprintf('times_%s.mat', ch_lbl));

    try
        if restore
            restore_channel(ch_lbl, spike_file, times_file);
            return;
        end

        if ~exist(spike_file, 'file')
            fprintf('  Channel %s: Spikes file not found (%s).\n', ch_lbl, spike_file);
            return;
        end

        SPK = load(spike_file);
        spikes_all = SPK.spikes_all;
        index_all  = SPK.index_all;
        index_all  = reshape(index_all, 1, []);

        n = numel(index_all);

        % ---- Build combined quarantine mask from selected/auto-detected masks ----
        masks_to_use = quarantine_masks;
        if isempty(masks_to_use)
            % auto-detect: use whichever of the valid masks exist in the file
            masks_to_use = valid_quarantine_masks(isfield(SPK, valid_quarantine_masks));
        else
            missing = masks_to_use(~isfield(SPK, masks_to_use));
            if ~isempty(missing)
                warning('  Channel %s: requested quarantine mask(s) not found, skipping: %s', ...
                    ch_lbl, strjoin(missing, ', '));
                masks_to_use = masks_to_use(isfield(SPK, masks_to_use));
            end
        end

        if isempty(masks_to_use)
            mask_pass_all = true(1, n);
            fprintf('  Channel %s: No quarantine masks found/selected; nothing to rescue.\n', ch_lbl);
        else
            mask_pass_all = true(1, n);
            for m = 1:length(masks_to_use)
                mvals = logical(reshape(SPK.(masks_to_use{m}), 1, []));
                if numel(mvals) ~= n
                    error('rescue_spikes:MaskLengthMismatch', ...
                        'Mask %s length (%d) does not match index_all length (%d) for channel %s.', ...
                        masks_to_use{m}, numel(mvals), n, ch_lbl);
                end
                mask_pass_all = mask_pass_all & mvals;
            end
        end

        % mask_quar: TRUE = spike fails at least one selected mask -> rescue candidate
        mask_quar = ~mask_pass_all;

        if ~any(mask_quar)
            fprintf('  Channel %s: No quarantined spikes (masks used: %s).\n', ...
                ch_lbl, format_mask_list(masks_to_use));
            return;
        end

        par = SPK.par;
        index  = SPK.index;
        spikes = SPK.spikes;

        par.amp_dir = amp_dir;
        par.pk_weight = peak_weight;

        % ---- Load times file / clustering info ----
        if exist(times_file, 'file')
            S = load(times_file);

            if ~isfield(S, 'spikes_pre_rescue')
                spikes_pre_rescue = S.spikes;
                index_pre_rescue = index;
                cluster_class_pre_rescue = S.cluster_class;
                save(times_file, 'spikes_pre_rescue', 'index_pre_rescue', ...
                     'cluster_class_pre_rescue', '-append');
            end

            cluster_class = S.cluster_class;
            if isfield(S, 'coeff')
                coeff = S.coeff;
            else
                coeff = 1:64; % Fallback if coeff missing
            end
            inspk_good = S.inspk;
        else
            % No times file: unclustered/multiunit. Treat all current spikes as
            % a single cluster and compute features locally.
            coeff = 1:64;
            inspk_good = local_wavelet_decomp(spikes);
            class_good_init = ones(size(spikes,1),1);
            cluster_class = [class_good_init, index(:)];
        end

        spikes_quar = spikes_all(mask_quar, :);
        index_quar  = index_all(mask_quar);

        % "Good"/clustered population = non-zero cluster assignments
        class_good_mask = cluster_class(:,1) ~= 0;
        class_good = cluster_class(class_good_mask, 1);
        inspk_good_classified  = inspk_good(class_good_mask, :);
        spikes_good_classified = spikes(class_good_mask, :);

        % ---- Feature extraction for quarantined spikes ----
        inspk_quar_full = local_wavelet_decomp(spikes_quar);
        inspk_quar = inspk_quar_full(:, coeff);

        % ---- Template matching ----
        par.sdnum = 3;
        class_quar = force_membership_wc(spikes_good_classified, class_good, spikes_quar, par);
        rescued_idx = find(class_quar ~= 0);

        if isempty(rescued_idx)
            fprintf('  Channel %s: No spikes rescued (masks used: %s).\n', ...
                ch_lbl, format_mask_list(masks_to_use));
        else
            fprintf('  Channel %s: Rescued %d/%d quarantined spikes (masks used: %s).\n', ...
                ch_lbl, numel(rescued_idx), numel(index_quar), format_mask_list(masks_to_use));
        end

        % ---- Merge rescued spikes with original clustered spikes ----
        spikes_rescued = spikes_quar(rescued_idx, :);
        index_rescued  = index_quar(rescued_idx);
        class_rescued  = class_quar(rescued_idx)';
        inspk_rescued  = inspk_quar(rescued_idx, :);

        index_combined  = [index(:); index_rescued(:)];
        spikes_combined = [spikes; spikes_rescued];
        class_combined  = [cluster_class(:,1); class_rescued(:)];
        inspk_combined  = [inspk_good; inspk_rescued];

        cluster_class_combined = zeros(length(class_combined), 2);
        cluster_class_combined(:,1) = class_combined;
        cluster_class_combined(:,2) = index_combined;

        [index_sorted, sort_idx] = sort(index_combined);
        spikes_sorted        = spikes_combined(sort_idx, :);
        cluster_class_sorted = cluster_class_combined(sort_idx, :);
        inspk_sorted         = inspk_combined(sort_idx, :);

        spikes = spikes_sorted;
        index = index_sorted;
        inspk = inspk_sorted;
        cluster_class = cluster_class_sorted;

        % ---- Build full-length rescue mask (aligned to index_all) ----
        rescue_mask = false(1, n);
        quar_indices_all = find(mask_quar);
        rescue_mask(quar_indices_all(rescued_idx)) = true;

        % Record which masks were used for this rescue pass
        quarantine_masks_used = masks_to_use;

        % ---- Save times file ----
        if exist(times_file, 'file')
            save(times_file, 'spikes', 'inspk', 'cluster_class', 'rescue_mask', ...
                 'quarantine_masks_used', '-append');
        else
            save(times_file, 'spikes', 'inspk', 'cluster_class', 'rescue_mask', ...
                 'quarantine_masks_used', 'par');
        end
        % Also persist per-spike rescue diagnostics in the times file
        save(times_file, 'class_quar', 'index_quar', 'rescued_idx', '-append');

        % ---- Save spikes file ----
        save(spike_file, 'spikes', 'index', 'rescue_mask', 'quarantine_masks_used', '-append');

    catch ME
        fprintf('  Channel %s: Error - %s\n', ch_lbl, ME.message);
        try
            report = getReport(ME, 'extended');
            fprintf('%s\n', report);
        catch
            fprintf('  (Could not get full report)\n');
        end
    end
end

function restore_channel(ch_lbl, spike_file, times_file)
    if exist(spike_file, 'file')
        vars_spk = load(spike_file);

        has_rescued_spikes = isfield(vars_spk, 'rescue_mask') && ~isempty(vars_spk.rescue_mask) && any(vars_spk.rescue_mask);

        if has_rescued_spikes
            rescued_timestamps = vars_spk.index_all(vars_spk.rescue_mask);
            to_remove_spk = ismember(vars_spk.index, rescued_timestamps);

            if any(to_remove_spk)
                vars_spk.spikes(to_remove_spk, :) = [];
                vars_spk.index(to_remove_spk) = [];
            end
            vars_spk.rescue_mask = false(size(vars_spk.index_all));
            save(spike_file, '-struct', 'vars_spk');
            fprintf('  Channel %s: Restored spikes file (removed %d rescued spikes).\n', ch_lbl, sum(to_remove_spk));
        end

        if exist(times_file, 'file')
            vars_times = load(times_file);

            if isfield(vars_times, 'spikes_pre_rescue')
                vars_times.spikes = vars_times.spikes_pre_rescue;
                vars_times.cluster_class = vars_times.cluster_class_pre_rescue;
                if isfield(vars_times, 'index_pre_rescue')
                    vars_times.index = vars_times.index_pre_rescue;
                end
                if isfield(vars_times, 'inspk') && isfield(vars_times, 'spikes_pre_rescue')
                    vars_times.inspk = vars_times.inspk(1:size(vars_times.spikes_pre_rescue,1), :);
                end
                fprintf('  Channel %s: Restored times file from backup.\n', ch_lbl);
            end

            fields_to_remove = {'spikes_quarantined', 'index_quarantined', 'class_quarantined', ...
                'class_quar', 'index_quar', 'rescued_idx', 'spikes_pre_rescue', 'index_pre_rescue', ...
                'cluster_class_pre_rescue', 'rescue_mask', 'quarantine_masks_used'};
            for f = 1:length(fields_to_remove)
                if isfield(vars_times, fields_to_remove{f})
                    vars_times = rmfield(vars_times, fields_to_remove{f});
                end
            end

            save(times_file, '-struct', 'vars_times');
        end

        if ~has_rescued_spikes
            if exist(times_file, 'file')
                fprintf('  Channel %s: No rescue mask found, but cleaned up times file.\n', ch_lbl);
            else
                fprintf('  Channel %s: No rescue mask found to restore.\n', ch_lbl);
            end
        end
    else
        fprintf('  Channel %s: Spikes file not found.\n', ch_lbl);
    end
end

function s = format_mask_list(masks)
    if isempty(masks)
        s = '(none)';
    else
        s = strjoin(masks, ', ');
    end
end

function inspk = local_wavelet_decomp(spikes)
    % Computes Haar wavelet coefficients for each spike
    nspk = size(spikes,1);
    L = size(spikes,2);
    scales = 4;
    cc = zeros(nspk, L);
    try
        spikes_l = reshape(spikes', numel(spikes), 1);
        if exist('wavedec', 'file')
            [c_l, l_wc] = wavedec(spikes_l, scales, 'haar');
        else
            [c_l, l_wc] = fix_wavedec(spikes_l, scales);
        end
        wv_c = [0; l_wc(1:end-1)];
        nc = wv_c / nspk;
        wccum = cumsum(wv_c);
        nccum = cumsum(nc);
        for cf = 2:length(nc)
            cc(:, nccum(cf-1)+1:nccum(cf)) = reshape(c_l(wccum(cf-1)+1:wccum(cf)), nc(cf), nspk)';
        end
    catch
        if exist('wavedec', 'file')
            for i = 1:nspk
                [c, ~] = wavedec(spikes(i,:), scales, 'haar');
                cc(i, 1:L) = c(1:L);
            end
        else
            for i = 1:nspk
                [c, ~] = fix_wavedec(spikes(i,:), scales);
                cc(i, 1:L) = c(1:L);
            end
        end
    end
    inspk = cc;
end