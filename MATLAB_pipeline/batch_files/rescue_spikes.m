function rescue_spikes(input, varargin)
% RESCUE_SPIKES  Reclassify quarantined spikes via template matching against
%                the clustered ("good") spike population.
%
% After Do_clustering, some spikes are quarantined by artifact/refractoriness
% masks. This function checks whether those spikes are close enough to an
% existing cluster template to be re-admitted ("rescued"). A rescue_mask
% aligned to index_all is saved so metrics can flag rescued spikes, and a
% full pre-rescue backup allows reverting with 'restore', true.
%
% Usage:
%   rescue_spikes(channels)
%   rescue_spikes(channels, 'sdnum', 3, 'template_type', 'center')
%   rescue_spikes(channels, 'restore', true)
%   rescue_spikes('all')
%   rescue_spikes(channels, 'folder', 'times_20260615_1623/ch333_merge[3_6]')
%
% Input:
%   input  - numeric vector of chan_ID values (matches NSx.chan_ID)
%             OR 'all' to process every times_*.mat in the active times folder
%             OR cell array of times_*.mat filenames
%
% Optional parameters (name-value):
%   'restore'       false     Revert times file to pre-rescue state
%   'sdnum'         3         Template radius in std-devs (same scale as Do_clustering)
%   'template_type' 'center'  Matching method: 'center','nn','mahal','ml'
%   'masks'         {}        Cell array of mask field names that define quarantine.
%                             Default auto-detects from: mask_nonart,
%                             mask_non_quarantine, mask_non_refract.
%                             mask_taskspks is never used.
%   'parallel'      false     Use parfor across channels
%   'folder'        ''        Override: path to times folder (or subfolder such
%                             as a merge folder). Relative to pwd or absolute.
%   'spikes_folder' ''        Override: path to spikes folder. Auto-detected
%                             from most-recent spikes* dir if not given.
%   'min_spikes'    10        Minimum number of quarantined spikes to attempt rescue.

% ---- Parse arguments -------------------------------------------------------
p = inputParser;
addParameter(p, 'restore',       false,    @islogical);
addParameter(p, 'sdnum',         3,        @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'template_type', 'center', @ischar);
addParameter(p, 'masks',         {},       @iscell);
addParameter(p, 'parallel',      false,    @islogical);
addParameter(p, 'folder_name',   '',       @ischar);   % relative to pwd, same as compute_metrics_batch
addParameter(p, 'spikes_folder', '',       @ischar);   % override spikes dir (default: root_dir/spikes*)
addParameter(p, 'min_spikes',    10,       @(x) isnumeric(x) && isscalar(x) && x >= 0);
parse(p, varargin{:});

restore          = p.Results.restore;
sdnum            = p.Results.sdnum;
template_type    = p.Results.template_type;
user_masks       = p.Results.masks;
do_parallel      = p.Results.parallel;
folder_name      = p.Results.folder_name;
spikes_folder_in = p.Results.spikes_folder;
min_spikes       = p.Results.min_spikes;

valid_masks = {'mask_nonart', 'mask_non_quarantine', 'mask_non_refract'};
if ~isempty(user_masks)
    bad = setdiff(user_masks, valid_masks);
    if ~isempty(bad)
        error('rescue_spikes: unknown mask name(s): %s\nValid: %s', ...
              strjoin(bad,', '), strjoin(valid_masks,', '));
    end
end

% ---- Resolve root and folder names (mirrors compute_metrics_batch) ---------
[~, current_dir_name] = fileparts(pwd);
root_dir = resolve_session_root();

% ---- Resolve times folder --------------------------------------------------
% Priority 1: user-supplied folder_name -> fullfile(pwd, folder_name)
% Priority 2: pwd itself if its name contains 'merge' or starts with 'times'
% Priority 3: auto-detect most recent times* dir inside pwd
if ~isempty(folder_name)
    active_times_dir = fullfile(pwd, folder_name);
    if ~exist(active_times_dir, 'dir')
        error('rescue_spikes: specified folder_name "%s" does not exist.', folder_name);
    end
    fprintf('Using user-specified folder: %s\n', active_times_dir);
elseif contains(current_dir_name, 'merge') || contains(current_dir_name, 'times_')
    active_times_dir = pwd;
else
    dates_times = dir(fullfile(pwd, 'times*'));
    dates_times = dates_times([dates_times.isdir]);
    if isempty(dates_times)
        error('rescue_spikes: No times* folders found in %s. Use ''folder_name'' to specify one.', pwd);
    end
    [~, idx_t] = max([dates_times.datenum]);
    active_times_dir = fullfile(pwd, dates_times(idx_t).name);
    fprintf('No folder specified. Auto-detecting most recent: %s\n', dates_times(idx_t).name);
end

% ---- Resolve spikes folder -------------------------------------------------
% Always from root_dir (not pwd) — spikes folder is session-level, not times-level.
% User override via 'spikes_folder' takes priority.
if ~isempty(spikes_folder_in)
    active_spikes_dir = spikes_folder_in;
    fprintf('Using user-specified spikes folder: %s\n', active_spikes_dir);
else
    dates_spikes = dir(fullfile(root_dir, 'spikes*'));
    dates_spikes = dates_spikes([dates_spikes.isdir]);
    if isempty(dates_spikes)
        error('rescue_spikes: No spikes* folders found in %s. Use ''spikes_folder'' to specify one.', root_dir);
    end
    [~, idx_s] = max([dates_spikes.datenum]);
    active_spikes_dir = fullfile(root_dir, dates_spikes(idx_s).name);
end

% ---- Build file list -------------------------------------------------------
file_list = resolve_file_list(input, active_times_dir);
if isempty(file_list)
    error('rescue_spikes: no times_*.mat files found for the given input.');
end
fprintf('rescue_spikes: %d file(s) to process.\n', numel(file_list));

% ---- Process ---------------------------------------------------------------
if do_parallel
    parfor k = 1:numel(file_list)
        process_one(file_list{k}, active_spikes_dir, restore, sdnum, ...
                    template_type, user_masks, valid_masks, min_spikes);
    end
else
    for k = 1:numel(file_list)
        process_one(file_list{k}, active_spikes_dir, restore, sdnum, ...
                    template_type, user_masks, valid_masks, min_spikes);
    end
end

if restore
    fprintf('rescue_spikes RESTORE complete.\n');
else
    fprintf('rescue_spikes complete.\n');
end
end

% ============================================================================
%  CORE PER-FILE LOGIC
% ============================================================================
function process_one(times_file, active_spikes_dir, restore, sdnum, ...
                     template_type, user_masks, valid_masks, min_spikes)

    [~, fname_noext] = fileparts(times_file);   % e.g. times_mRAMY04_raw_333
    % Strip leading 'times_' to get the channel label used in spikes folder
    ch_lbl = regexprep(fname_noext, '^times_', '');   % e.g. mRAMY04_raw_333
    spike_file = fullfile(active_spikes_dir, [ch_lbl '_spikes.mat']);

    if restore
        do_restore(times_file, spike_file, ch_lbl);
        return;
    end

    % ---- Guard: files must exist -------------------------------------------
    if ~exist(times_file, 'file')
        fprintf('  [%s] times file not found, skipping.\n', ch_lbl);
        return;
    end
    if ~exist(spike_file, 'file')
        fprintf('  [%s] spikes file not found (%s), skipping.\n', ch_lbl, spike_file);
        return;
    end

    % ---- Load spikes file (has spikes_all / index_all / masks) -------------
    fprintf('  [%s] Loading spikes file...\n', ch_lbl);
    SPK = load(spike_file);

    if ~isfield(SPK, 'spikes_all') || ~isfield(SPK, 'index_all')
        fprintf('  [%s] spikes file missing spikes_all/index_all, skipping.\n', ch_lbl);
        return;
    end

    spikes_all = double(SPK.spikes_all);
    index_all  = reshape(double(SPK.index_all), 1, []);
    n_all      = numel(index_all);

    % ---- Build quarantine mask from spikes file ----------------------------
    masks_to_use = user_masks;
    if isempty(masks_to_use)
        masks_to_use = valid_masks(isfield(SPK, valid_masks));
    else
        missing = masks_to_use(~isfield(SPK, masks_to_use));
        if ~isempty(missing)
            warning('  [%s] mask(s) not in spikes file, ignoring: %s', ...
                    ch_lbl, strjoin(missing,', '));
            masks_to_use = masks_to_use(isfield(SPK, masks_to_use));
        end
    end

    if isempty(masks_to_use)
        fprintf('  [%s] No quarantine masks found. Nothing to rescue.\n', ch_lbl);
        return;
    end

    % mask_pass: TRUE = spike passes ALL selected masks (is "good")
    mask_pass = true(1, n_all);
    for mi = 1:numel(masks_to_use)
        mv = logical(reshape(SPK.(masks_to_use{mi}), 1, []));
        if numel(mv) ~= n_all
            error('  [%s] mask %s length (%d) != index_all length (%d)', ...
                  ch_lbl, masks_to_use{mi}, numel(mv), n_all);
        end
        mask_pass = mask_pass & mv;
    end
    mask_quar = ~mask_pass;   % TRUE = quarantined -> rescue candidate

    n_quar = sum(mask_quar);
    if n_quar < min_spikes
        fprintf('  [%s] Only %d quarantined spikes (min %d). Skipping.\n', ...
                ch_lbl, n_quar, min_spikes);
        return;
    end
    fprintf('  [%s] %d quarantined spikes (masks: %s)\n', ...
            ch_lbl, n_quar, strjoin(masks_to_use, ', '));

    % ---- Load times file ---------------------------------------------------
    fprintf('  [%s] Loading times file...\n', ch_lbl);
    T = load(times_file);

    if ~isfield(T, 'cluster_class') || ~isfield(T, 'spikes')
        fprintf('  [%s] times file missing cluster_class or spikes, skipping.\n', ch_lbl);
        return;
    end

    % ---- Backup: save pre-rescue state (only on FIRST rescue pass) ---------
    if ~isfield(T, 'cluster_class_pre_rescue')
        fprintf('  [%s] Saving pre-rescue backup...\n', ch_lbl);
        cluster_class_pre_rescue = T.cluster_class;   
        spikes_pre_rescue        = T.spikes;          
        inspk_pre_rescue         = T.inspk;           
        save(times_file, ...
             'cluster_class_pre_rescue', 'spikes_pre_rescue', 'inspk_pre_rescue', ...
             '-append');
    else
        fprintf('  [%s] Pre-rescue backup already exists; running additional pass.\n', ch_lbl);
    end

    % ---- Clustered ("good") population -------------------------------------
    cluster_class = T.cluster_class;
    spikes_good   = T.spikes;
    inspk_good    = T.inspk;

    % Use coeff from times file if available (may differ from spikes file)
    if isfield(T, 'coeff')
        coeff = T.coeff;
    elseif isfield(SPK, 'coeff')
        coeff = SPK.coeff;
    else
        coeff = 1:min(64, size(inspk_good, 2));
    end

    % Only non-zero clusters participate as templates
    good_mask   = cluster_class(:,1) ~= 0;
    class_good  = cluster_class(good_mask, 1);
    spikes_tmpl = spikes_good(good_mask, :);
    inspk_tmpl  = inspk_good(good_mask, :);

    if isempty(class_good)
        fprintf('  [%s] No non-zero clusters to build templates from. Skipping.\n', ch_lbl);
        return;
    end

    % ---- Quarantined spike waveforms / features ----------------------------
    spikes_quar = spikes_all(mask_quar, :);
    index_quar  = index_all(mask_quar);

    % Feature extraction (wavelet) matching dimensionality of inspk_tmpl
    inspk_quar_full = local_wavelet_decomp(spikes_quar);
    n_coeff = size(inspk_tmpl, 2);
    if size(inspk_quar_full, 2) >= n_coeff
        inspk_quar = inspk_quar_full(:, 1:n_coeff);
    else
        % Pad with zeros if shorter (should not normally happen)
        inspk_quar = [inspk_quar_full, zeros(size(inspk_quar_full,1), n_coeff - size(inspk_quar_full,2))];
    end

    % ---- Template matching -------------------------------------------------
    par_tmpl = struct();
    par_tmpl.template_type   = template_type;
    par_tmpl.template_sdnum  = sdnum;
    par_tmpl.template_k      = 10;
    par_tmpl.template_k_min  = 5;
    par_tmpl.sdnum           = sdnum;

    fprintf('  [%s] Template matching (%s, sdnum=%.1f)...\n', ch_lbl, template_type, sdnum);
    class_quar = force_membership_wc(spikes_tmpl, class_good, spikes_quar, par_tmpl);

    rescued_local = find(class_quar ~= 0);   % indices into spikes_quar
    n_rescued = numel(rescued_local);
    fprintf('  [%s] Rescued %d / %d quarantined spikes.\n', ch_lbl, n_rescued, n_quar);

    % ---- Build rescue_mask aligned to index_all ----------------------------
    % rescue_mask(i) = true means index_all(i) was quarantined AND rescued.
    rescue_mask = false(1, n_all);
    quar_positions = find(mask_quar);           % positions in index_all
    rescue_mask(quar_positions(rescued_local)) = true;

    if n_rescued == 0
        % Still save the rescue_mask (all false) and metadata
        quarantine_masks_used = masks_to_use;   
        rescue_meta = make_rescue_meta(sdnum, template_type, masks_to_use, n_quar, 0);  
        save(times_file, 'rescue_mask', 'quarantine_masks_used', 'rescue_meta', '-append');
        return;
    end

    % ---- Merge rescued spikes into times file arrays -----------------------
    spikes_resc = spikes_quar(rescued_local, :);
    index_resc  = index_quar(rescued_local);
    class_resc  = class_quar(rescued_local)';
    inspk_resc  = inspk_quar(rescued_local, :);

    % Append to existing good population
    index_combined  = [cluster_class(:,2);  index_resc(:)];
    spikes_combined = [spikes_good;          spikes_resc];
    class_combined  = [cluster_class(:,1);  class_resc(:)];
    inspk_combined  = [inspk_good;           inspk_resc];

    % Sort by spike time
    [~, sort_idx]   = sort(index_combined);
    spikes_new      = spikes_combined(sort_idx, :);
    inspk_new       = inspk_combined(sort_idx, :);
    cluster_class_new = [class_combined(sort_idx), index_combined(sort_idx)];

    % ---- Metadata ----------------------------------------------------------
    quarantine_masks_used = masks_to_use;   
    rescue_meta = make_rescue_meta(sdnum, template_type, masks_to_use, n_quar, n_rescued);  

    % ---- Save times file ---------------------------------------------------
    spikes = spikes_new;           
    inspk  = inspk_new;            
    cluster_class = cluster_class_new;  

    save(times_file, ...
         'spikes', 'inspk', 'cluster_class', ...
         'rescue_mask', 'quarantine_masks_used', 'rescue_meta', ...
         '-append');

    fprintf('  [%s] Done. %d rescued spikes added to times file.\n', ch_lbl, n_rescued);
end

% ============================================================================
%  RESTORE
% ============================================================================
function do_restore(times_file, spike_file, ch_lbl)
    if ~exist(times_file, 'file')
        fprintf('  [%s] times file not found, nothing to restore.\n', ch_lbl);
        return;
    end

    T = load(times_file);

    if ~isfield(T, 'cluster_class_pre_rescue')
        fprintf('  [%s] No pre-rescue backup found; already clean.\n', ch_lbl);
        return;
    end

    % Restore core arrays from backup
    T.cluster_class = T.cluster_class_pre_rescue;
    T.spikes        = T.spikes_pre_rescue;
    T.inspk         = T.inspk_pre_rescue;

    % Remove all rescue-related fields
    rescue_fields = {'cluster_class_pre_rescue', 'spikes_pre_rescue', 'inspk_pre_rescue', ...
                     'rescue_mask', 'quarantine_masks_used', 'rescue_meta', ...
                     'class_quar', 'index_quar', 'rescued_idx'};
    for fi = 1:numel(rescue_fields)
        if isfield(T, rescue_fields{fi})
            T = rmfield(T, rescue_fields{fi});
        end
    end

    save(times_file, '-struct', 'T');
    fprintf('  [%s] Restored to pre-rescue state.\n', ch_lbl);

    % Clean rescue_mask from spikes file too
    if exist(spike_file, 'file')
        S = load(spike_file);
        if isfield(S, 'rescue_mask')
            S = rmfield(S, 'rescue_mask');
            save(spike_file, '-struct', 'S');
            fprintf('  [%s] Cleared rescue_mask from spikes file.\n', ch_lbl);
        end
    end
end

% ============================================================================
%  FILE LIST RESOLVER  (mirrors Do_clustering / compute_metrics_batch style)
% ============================================================================
function file_list = resolve_file_list(input, times_dir)
    file_list = {};

    if ischar(input) || isstring(input)
        input = char(input);
        if strcmp(input, 'all')
            d = dir(fullfile(times_dir, 'times_*.mat'));
            file_list = cellfun(@(n) fullfile(times_dir, n), {d.name}, 'UniformOutput', false);
            return;
        end
        % Single filename or absolute path
        if exist(input, 'file')
            file_list = {input};
        else
            candidate = fullfile(times_dir, input);
            if exist(candidate, 'file')
                file_list = {candidate};
            end
        end
        return;
    end

    if iscell(input)
        % Cell of filenames or absolute paths
        for k = 1:numel(input)
            f = input{k};
            if exist(f, 'file')
                file_list{end+1} = f;  
            else
                candidate = fullfile(times_dir, f);
                if exist(candidate, 'file')
                    file_list{end+1} = candidate;  
                else
                    warning('rescue_spikes: file not found: %s', f);
                end
            end
        end
        return;
    end

    if isnumeric(input)
        % Channel IDs — match against times_*.mat filenames
        d = dir(fullfile(times_dir, 'times_*.mat'));
        all_names = {d.name};
        for ci = input(:)'
            pattern = sprintf('%d', ci);
            matched = false;
            for k = 1:numel(all_names)
                nm = all_names{k};
                % Extract trailing digits before .mat
                tok = regexp(nm, '(\d+)\.mat$', 'tokens', 'once');
                if ~isempty(tok) && str2double(tok{1}) == ci
                    file_list{end+1} = fullfile(times_dir, nm);  
                    matched = true;
                    break;
                end
            end
            if ~matched
                warning('rescue_spikes: no times_*.mat found for channel %d', ci);
            end
        end
        return;
    end

    error('rescue_spikes: unrecognized input type.');
end

% ============================================================================
%  HELPERS
% ============================================================================
function meta = make_rescue_meta(sdnum, template_type, masks, n_quar, n_rescued)
    meta.timestamp     = datestr(now, 'yyyy-mm-dd HH:MM:SS');
    meta.sdnum         = sdnum;
    meta.template_type = template_type;
    meta.masks_used    = masks;
    meta.n_quarantined = n_quar;
    meta.n_rescued     = n_rescued;
end

function inspk = local_wavelet_decomp(spikes)
% Haar wavelet decomposition matching the feature extraction in Do_features.
    nspk = size(spikes,1);
    L    = size(spikes,2);
    scales = 4;
    cc = zeros(nspk, L);
    try
        spikes_l = reshape(spikes', numel(spikes), 1);
        if exist('wavedec','file')
            [c_l, l_wc] = wavedec(spikes_l, scales, 'haar');
        else
            [c_l, l_wc] = fix_wavedec(spikes_l, scales);
        end
        wv_c  = [0; l_wc(1:end-1)];
        nc    = wv_c / nspk;
        wccum = cumsum(wv_c);
        nccum = cumsum(nc);
        for cf = 2:length(nc)
            cc(:, nccum(cf-1)+1:nccum(cf)) = ...
                reshape(c_l(wccum(cf-1)+1:wccum(cf)), nc(cf), nspk)';
        end
    catch
        % Fallback: per-spike loop
        for i = 1:nspk
            if exist('wavedec','file')
                [c, ~] = wavedec(spikes(i,:), scales, 'haar');
            else
                [c, ~] = fix_wavedec(spikes(i,:), scales);
            end
            cc(i, 1:min(L,numel(c))) = c(1:min(L,numel(c)));
        end
    end
    inspk = cc;
end