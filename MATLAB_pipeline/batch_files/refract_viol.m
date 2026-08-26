function refract_viol(channels,varargin)
    % Function: refract_viol
    % Description: Filters spikes that violate the refractory period.
    %              The refractory parameters (ref_ms or ref) mimic those used
    %              in amp_detect.m. The resulting mask is combined with existing
    %              masks for cumulative filtering.
    % Channels: The list of microelectrode channels (channel IDs) to process.
    % Optional name-value args:
    %   'keep_strategy' (default 'none') - 'first', 'last', or 'none' (flag whole chain).
    %   'use_parallel'  (default false)  - run channels with parfor instead of for.
    %   'allow_stitch_fallback' (default false) - if a channel's spikes.mat has
    %       no 'refract_chains' (i.e. Get_spikes never saw a continuous signal
    %       for it, e.g. the with_spikes/pre-detected-spikes path), setting
    %       this true reconstructs an approximate chain trace by stitching
    %       together the overlapping spikes_all windows instead (lower
    %       fidelity than a real continuous slice - no padding beyond
    %       w_pre/w_post is possible since spikes_all windows don't extend
    %       past that). Default false: such channels are left without
    %       refract_chains and a warning explains how to get them.
    %
    % refract_chains, when present, is a struct array (one entry per
    % violation chain) with the chain's continuous trace, its absolute start
    % time, sample rate, and the member indices into index_all/spikes_all.
    % It is normally saved by Get_spikes (from the real continuous filtered
    % signal, with refract_chain_pad extra samples on each side); this
    % function only masks out the violating spikes and, when
    % 'allow_stitch_fallback' is set, provides the degraded fallback above.

    p = inputParser;
    addParameter(p, 'keep_strategy', 'none', @ischar);
    addParameter(p, 'use_parallel', false, @islogical);
    addParameter(p, 'allow_stitch_fallback', false, @islogical);
    parse(p, varargin{:});

    keep_strategy = p.Results.keep_strategy;
    use_parallel = p.Results.use_parallel;
    allow_stitch_fallback = p.Results.allow_stitch_fallback;

    refract_viol_tic = tic;

    load('NSx','NSx');
    % Filter NSx structure to include only the specified channels
    NSx_proc = NSx(ismember(cell2mat({NSx.chan_ID}),channels));

    num_channels_proc = length(NSx_proc);
    fprintf('Starting refractory violation check on %d channels (Parallel: %d)...\n', num_channels_proc, use_parallel);

    dates = dir(fullfile(pwd, 'spikes*'));
    dates = dates([dates.isdir]);
    if isempty(dates), error('No spikes folders found.'); end
    [~, idx] = max([dates.datenum]);
    active_spikes_dir = fullfile(pwd, dates(idx).name);

    chan_lbls = {NSx_proc.output_name};

    if use_parallel
        parfor k = 1:num_channels_proc
            process_refract_channel(chan_lbls{k}, active_spikes_dir, keep_strategy, allow_stitch_fallback);
        end
    else
        for k = 1:num_channels_proc
            process_refract_channel(chan_lbls{k}, active_spikes_dir, keep_strategy, allow_stitch_fallback);
        end
    end

    refract_viol_toc = toc(refract_viol_tic);
    fprintf("refract_viol DONE in %s seconds.\n", num2str(refract_viol_toc, '%2.2f'));
end

% --- LOCAL HELPER FUNCTION FOR LOOP BODY ---
function process_refract_channel(ch_lbl, active_spikes_dir, keep_strategy, allow_stitch_fallback)
    % Target the file inside our locked directory
    spike_file = fullfile(active_spikes_dir, sprintf('%s_spikes.mat', ch_lbl));

    try
        % fprintf('ch.%d/%d %s: loading %s\n', k, num_channels_proc, ch_lbl, spike_file);
        SPK = load(spike_file);

        % Load full spike set
        if isfield(SPK,'spikes_all')
            spikes_all = SPK.spikes_all;
            index_all  = SPK.index_all(:);
        else
            spikes_all = SPK.spikes;
            index_all  = SPK.index(:);
        end

        par = SPK.par;

        % Determine refractory period in milliseconds
        if isfield(par, 'ref_ms')
            ref_val = par.ref_ms;
        elseif isfield(par, 'ref')
            % par.ref was historically in samples, convert to ms
            ref_val = par.ref / (par.sr / 1000);
        else
            ref_val = 1.5; % Default fallback to 1.5ms
        end

        % refract_chains is normally already saved by Get_spikes, sliced
        % straight from the real continuous filtered signal it had on hand
        % during detection. This function never rebuilds those - it only
        % masks out the violating spikes. The stitched-from-spikes_all
        % fallback below only runs for channels that never got a native
        % refract_chains (no continuous signal was available at detection
        % time), and only when explicitly opted into.
        has_native_chains = isfield(SPK, 'refract_chains');
        chains_to_save = false;

        if length(index_all) > 1
            % Find gaps between consecutive spikes
            gaps = diff(index_all);
            in_chain_gap = gaps < ref_val; % True if gap violates refractory window

            % Identify the starts and ends of chains
            % We pad with false to catch chains at the very edges of the array
            ext_gap = [false; in_chain_gap; false];

            % Rise in True indicates a chain started; Fall indicates it ended
            chain_starts = find(diff(ext_gap) == 1);
            chain_ends   = find(diff(ext_gap) == -1);

            if ~has_native_chains
                if allow_stitch_fallback
                    % Fallback: approximate each chain as a single continuous,
                    % unaligned trace by stitching the raw spikes_all windows
                    % together (no per-spike re-centering). Each member's
                    % window is a contiguous slice of the same filtered
                    % signal; since the chain-defining gap is always <
                    % ref_val and ref_val < the window width, only the new
                    % trailing samples each later member contributes need
                    % appending. Lower fidelity than a native trace: spikes_all
                    % windows don't extend past w_pre/w_post, so no
                    % refract_chain_pad-style padding is possible here.
                    refract_chains = struct('chain_index', {}, 'trace', {}, 't0_ms', {}, 'sr', {});
                    sr = par.sr;
                    wlen = size(spikes_all, 2);
                    for c = 1:numel(chain_starts)
                        members = chain_starts(c):chain_ends(c);
                        trace = spikes_all(members(1), :);
                        for m = 2:numel(members)
                            gap_samp = round((index_all(members(m)) - index_all(members(m-1))) * sr / 1000);
                            gap_samp = max(1, min(gap_samp, wlen));
                            trace = [trace, spikes_all(members(m), wlen-gap_samp+1:end)];
                        end
                        refract_chains(c).chain_index = members;
                        refract_chains(c).trace   = trace;
                        refract_chains(c).t0_ms   = index_all(members(1)) - par.w_pre * 1000/sr;
                        refract_chains(c).sr      = sr;
                    end
                    chains_to_save = true;
                elseif ~isempty(chain_starts)
                    warning('RefractViol:NoContinuousChains', ...
                        ['Channel %s: %d refractory violation chain(s) found but no refract_chains ' ...
                         'in %s (no continuous signal was available when spikes were detected, e.g. ' ...
                         'the pre-detected-spikes path in Get_spikes). Re-run Get_spikes on this channel ' ...
                         'from raw data to get real continuous-trace chains, or call refract_viol with ' ...
                         '''allow_stitch_fallback'', true to reconstruct an approximate stitched trace instead.'], ...
                        ch_lbl, numel(chain_starts), spike_file);
                end
            end

            % Allocate our violation wmask
            mask_refract = false(size(index_all));

            switch lower(keep_strategy)
                case 'first'
                    % Keep the anchor (chain_starts), flag all subsequent spikes
                    for c = 1:length(chain_starts)
                        mask_refract(chain_starts(c) + 1 : chain_ends(c)) = true;
                    end

                case 'last'
                    % Flag all prior spikes, keep the final element (chain_ends)
                    for c = 1:length(chain_starts)
                        mask_refract(chain_starts(c) : chain_ends(c) - 1) = true;
                    end

                otherwise % 'none' / default conservative choice
                    % Flag every single spike involved in the chain
                    for c = 1:length(chain_starts)
                        mask_refract(chain_starts(c) : chain_ends(c)) = true;
                    end
            end
        else
            mask_refract = false(size(index_all));
        end

        mask_non_refract = ~mask_refract;

        % Load existing masks to create a cumulative mask
        if isfield(SPK, 'mask_nonart')
            mask_nonart = SPK.mask_nonart;
        else
            mask_nonart = true(size(index_all));
        end

        if isfield(SPK, 'mask_non_quarantine')
            mask_non_quarantine = SPK.mask_non_quarantine;
        else
            mask_non_quarantine = true(size(index_all));
        end

        if isfield(SPK, 'mask_taskspks')
            mask_taskspks = reshape(SPK.mask_taskspks, 1, []);
        else
            mask_taskspks = true(length(index_all),1);
        end

        % Normalize every mask to the same orientation regardless of
        % source (loaded from file vs. default fallback) - loaded masks
        % and fallback defaults are not guaranteed to agree on row vs.
        % column, and mixing them silently broadcasts to an NxN matrix
        % instead of erroring, which then breaks spikes_all(mask_tot,:).
        mask_nonart          = reshape(mask_nonart, 1, []);
        mask_non_quarantine  = reshape(mask_non_quarantine, 1, []);
        mask_taskspks        = reshape(mask_taskspks, 1, []);
        mask_non_refract     = reshape(mask_non_refract, 1, []);

        % Apply previous and new mask
        mask_tot = mask_nonart & mask_non_quarantine & mask_taskspks & mask_non_refract;

        spikes = spikes_all(mask_tot, :);
        index = index_all(mask_tot);

        index = reshape(index, 1, []);
        index_all = reshape(index_all, 1, []);
        mask_non_refract = reshape(mask_non_refract, 1, []);

        % Single-quoted variable names for safe saving (parfor-compatible)
        save_vars = {'index', 'spikes', 'index_all', 'spikes_all', 'par', 'mask_non_refract'};
        if chains_to_save
            save_vars = [save_vars, {'refract_chains'}];
        end
        save(spike_file, save_vars{:}, '-append');

        if has_native_chains
            chain_note = sprintf('%d native chains (from Get_spikes)', numel(SPK.refract_chains));
        elseif chains_to_save
            chain_note = sprintf('%d stitched-fallback chains', numel(refract_chains));
        else
            chain_note = 'no chains saved';
        end
        fprintf('Channel %s: %d/%d spikes flagged as refractory violations (%.2f%%), %s\n', ...
            ch_lbl, sum(mask_refract), length(index_all), (sum(mask_refract) / length(index_all)) * 100, chain_note);

    catch ME
        fprintf('  -> FAILED to process channel %s: %s\n', ch_lbl, ME.message);
    end
end