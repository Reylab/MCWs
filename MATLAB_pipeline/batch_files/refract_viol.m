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

    p = inputParser;
    addParameter(p, 'keep_strategy', 'none', @ischar);
    addParameter(p, 'use_parallel', false, @islogical);
    parse(p, varargin{:});

    keep_strategy = p.Results.keep_strategy;
    use_parallel = p.Results.use_parallel;

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
            process_refract_channel(chan_lbls{k}, active_spikes_dir, keep_strategy);
        end
    else
        for k = 1:num_channels_proc
            process_refract_channel(chan_lbls{k}, active_spikes_dir, keep_strategy);
        end
    end

    refract_viol_toc = toc(refract_viol_tic);
    fprintf("refract_viol DONE in %s seconds.\n", num2str(refract_viol_toc, '%2.2f'));
end

% --- LOCAL HELPER FUNCTION FOR LOOP BODY ---
function process_refract_channel(ch_lbl, active_spikes_dir, keep_strategy)
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

        % Vectorized Refractory Chain Flagging
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

        mask_non_refract = ~mask_refract';

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
            mask_taskspks = true(1, length(index_all));
        end

        % Apply previous and new mask
        mask_tot = mask_nonart & mask_non_quarantine & mask_taskspks & mask_non_refract;

        spikes = spikes_all(mask_tot, :);
        index = index_all(mask_tot);

        index = reshape(index, 1, []);
        index_all = reshape(index_all, 1, []);
        mask_non_refract = reshape(mask_non_refract, 1, []);

        % Single-quoted variable names for safe saving (parfor-compatible)
        save(spike_file, ...
             'index', 'spikes', 'index_all', 'spikes_all', 'par', ...
             'mask_non_refract', '-append');

        fprintf('Channel %s: %d/%d spikes flagged as refractory violations (%.2f%%)\n', ...
            ch_lbl, sum(mask_refract),length(index_all), (sum(mask_refract) / length(index_all)) * 100);

    catch ME
        fprintf('  -> FAILED to process channel %s: %s\n', ch_lbl, ME.message);
    end
end