function apply_masks_to_times(input, varargin)
% APPLY_MASKS_TO_TIMES  Reduce a times file's clustering to the spikes that
% survive the spikes-file artifact/QC masks, WITHOUT re-running feature
% extraction or SPC. The cluster identities are kept exactly as they were;
% only the masked-out spikes are dropped from cluster_class / spikes / inspk.
%
% This is the mechanism that lets validate_mask_effects compare class
% responses across "no-mask", "masked" and "rescued" states while the
% feature space (and therefore the clustering) is held fixed.
%
% Usage:
%   apply_masks_to_times(channels)
%   apply_masks_to_times('all')
%   apply_masks_to_times(channels, 'times_dir', T, 'spikes_dir', S)
%   apply_masks_to_times(channels, 'masks', {'mask_non_quarantine','mask_non_refract'})
%   apply_masks_to_times(channels, 'restore', true)   % undo, from *_premask backup
%
% Input:  numeric chan_ID vector | 'all' | cell array of times_*.mat names.
%
% Name-value:
%   'times_dir'  ''   times folder (default: newest times* under pwd, or pwd
%                     if its name starts with 'times')
%   'spikes_dir' ''   spikes folder (default: newest spikes* under
%                     resolve_session_root())
%   'masks'      {'mask_non_quarantine','mask_non_refract'}  spikes-file mask
%                     fields that must ALL be true for a spike to survive.
%   'restore'    false  revert cluster_class / spikes / inspk from the
%                     *_premask backup and delete the backup fields.

    p = inputParser;
    addParameter(p, 'times_dir',  '', @ischar);
    addParameter(p, 'spikes_dir', '', @ischar);
    addParameter(p, 'masks', {'mask_non_quarantine','mask_non_refract'}, @iscell);
    addParameter(p, 'restore', false, @islogical);
    parse(p, varargin{:});
    masks   = p.Results.masks;
    restore = p.Results.restore;

    root = resolve_session_root();

    % ---- times folder --------------------------------------------------
    times_dir = p.Results.times_dir;
    if isempty(times_dir)
        [~, cur] = fileparts(pwd);
        if startsWith(cur, 'times')
            times_dir = pwd;
        else
            d = dir(fullfile(pwd, 'times*')); d = d([d.isdir]);
            if isempty(d), d = dir(fullfile(root, 'times*')); d = d([d.isdir]); end
            if isempty(d), error('apply_masks_to_times: no times* folder found.'); end
            [~, i] = max([d.datenum]);
            times_dir = fullfile(d(i).folder, d(i).name);
        end
    end

    % ---- spikes folder -----------------------------------------------
    spikes_dir = p.Results.spikes_dir;
    if isempty(spikes_dir)
        d = dir(fullfile(root, 'spikes*')); d = d([d.isdir]);
        if isempty(d), d = dir(fullfile(pwd, 'spikes*')); d = d([d.isdir]); end
        if isempty(d), error('apply_masks_to_times: no spikes* folder found.'); end
        [~, i] = max([d.datenum]);
        spikes_dir = fullfile(d(i).folder, d(i).name);
    end
    fprintf('apply_masks_to_times: times=%s  spikes=%s\n', times_dir, spikes_dir);

    % ---- resolve file list -----------------------------------------
    load(fullfile(root, 'NSx.mat'), 'NSx');
    files = local_file_list(input, times_dir, NSx);
    if isempty(files), error('apply_masks_to_times: no times_*.mat matched the input.'); end

    for k = 1:numel(files)
        tf = files{k};
        [~, base] = fileparts(tf);
        lbl = regexprep(base, '^times_', '');
        sf  = fullfile(spikes_dir, [lbl '_spikes.mat']);
        if restore
            do_restore(tf, lbl);
            continue;
        end
        if ~exist(sf, 'file')
            fprintf('  [%s] spikes file missing (%s) - skipped.\n', lbl, sf);
            continue;
        end
        process_one(tf, sf, lbl, masks);
    end
    if restore
        fprintf('apply_masks_to_times RESTORE complete.\n');
    else
        fprintf('apply_masks_to_times complete.\n');
    end
end

% ====================================================================
function process_one(times_file, spike_file, lbl, masks)
    T = load(times_file);
    if ~isfield(T, 'cluster_class')
        fprintf('  [%s] no cluster_class - skipped.\n', lbl); return;
    end
    if isfield(T, 'cluster_class_premask')
        fprintf('  [%s] already has cluster_class_premask - skipped (restore first).\n', lbl);
        return;
    end
    S = load(spike_file);
    if ~isfield(S, 'index_all')
        fprintf('  [%s] spikes file has no index_all - skipped.\n', lbl); return;
    end

    idx_all  = reshape(double(S.index_all), 1, []);
    keep_all = true(1, numel(idx_all));
    used = {};
    for m = 1:numel(masks)
        mf = masks{m};
        if isfield(S, mf)
            mv = logical(reshape(S.(mf), 1, []));
            if numel(mv) ~= numel(idx_all)
                error('  [%s] mask %s length (%d) ~= index_all (%d)', ...
                      lbl, mf, numel(mv), numel(idx_all));
            end
            keep_all = keep_all & mv;
            used{end+1} = mf;
        else
            fprintf('  [%s] mask %s not in spikes file - ignored.\n', lbl, mf);
        end
    end
    if isempty(used)
        fprintf('  [%s] none of the requested masks present - nothing to do.\n', lbl);
        return;
    end

    cc = T.cluster_class;
    [tf, loc] = ismember(cc(:,2), idx_all);
    row_keep = false(size(cc,1), 1);
    row_keep(tf) = keep_all(loc(tf));
    if any(~tf)
        fprintf('  [%s] %d cluster rows not found in index_all (kept as-is).\n', lbl, nnz(~tf));
        row_keep(~tf) = true;
    end

    n0 = size(cc,1);
    has_spk = isfield(T,'spikes') && size(T.spikes,1) == n0;
    has_ins = isfield(T,'inspk')  && size(T.inspk,1)  == n0;

    % ---- backup ----
    cluster_class_premask = T.cluster_class;
    save(times_file, 'cluster_class_premask', '-append');
    if has_spk, spikes_premask = T.spikes; save(times_file,'spikes_premask','-append'); end
    if has_ins, inspk_premask  = T.inspk;  save(times_file,'inspk_premask','-append');  end
    mask_reduce_meta = struct('timestamp', char(datetime('now','Format','yyyy-MM-dd HH:mm:ss')), ...
                              'masks_used', {used}, 'n_before', n0, ...
                              'n_after', nnz(row_keep));
    save(times_file, 'mask_reduce_meta', '-append');

    % ---- reduce ----
    cluster_class = cc(row_keep, :);
    save(times_file, 'cluster_class', '-append');
    if has_spk, spikes = T.spikes(row_keep, :); save(times_file,'spikes','-append'); end
    if has_ins, inspk  = T.inspk(row_keep, :);  save(times_file,'inspk','-append');  end

    fprintf('  [%s] %s: %d -> %d clustered spikes (dropped %d)\n', ...
            lbl, strjoin(used,'+'), n0, nnz(row_keep), nnz(~row_keep));
end

% ====================================================================
function do_restore(times_file, lbl)
    if ~exist(times_file, 'file'), return; end
    T = load(times_file);
    if ~isfield(T, 'cluster_class_premask')
        fprintf('  [%s] no premask backup - nothing to restore.\n', lbl); return;
    end
    T.cluster_class = T.cluster_class_premask;
    if isfield(T,'spikes_premask'), T.spikes = T.spikes_premask; end
    if isfield(T,'inspk_premask'),  T.inspk  = T.inspk_premask;  end
    for f = {'cluster_class_premask','spikes_premask','inspk_premask','mask_reduce_meta'}
        if isfield(T, f{1}), T = rmfield(T, f{1}); end
    end
    save(times_file, '-struct', 'T');
    fprintf('  [%s] restored from premask backup.\n', lbl);
end

% ====================================================================
function file_list = local_file_list(input, times_dir, NSx)
    file_list = {};
    if ischar(input) || isstring(input)
        input = char(input);
        if strcmpi(input, 'all')
            d = dir(fullfile(times_dir, 'times_*.mat'));
            file_list = cellfun(@(n) fullfile(times_dir, n), {d.name}, 'UniformOutput', false);
        elseif exist(input, 'file')
            file_list = {input};
        elseif exist(fullfile(times_dir, input), 'file')
            file_list = {fullfile(times_dir, input)};
        end
        return;
    end
    if iscell(input)
        for i = 1:numel(input)
            c = input{i};
            if exist(c,'file'), file_list{end+1} = c;
            elseif exist(fullfile(times_dir,c),'file'), file_list{end+1} = fullfile(times_dir,c);
            end
        end
        return;
    end
    if isnumeric(input)
        for ci = input(:)'
            pos = find(arrayfun(@(x) x.chan_ID == ci, NSx), 1);
            if isempty(pos)
                fprintf('apply_masks_to_times: chan %d not in NSx.\n', ci); continue;
            end
            cand = fullfile(times_dir, sprintf('times_%s.mat', NSx(pos).output_name));
            if exist(cand, 'file'), file_list{end+1} = cand;
            else, fprintf('apply_masks_to_times: %s not found.\n', cand);
            end
        end
        return;
    end
    error('apply_masks_to_times: unrecognized input type.');
end
