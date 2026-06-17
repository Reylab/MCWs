function varargout = merge_clusters(channels, merge_list, varargin)
% MERGE_CLUSTERS - Merge clusters for targeted channels using NSx mapping.
%
% Usage (single group):
%   merge_clusters(channels, [3 5], 'folder', '')
%       Merges clusters 3 and 5 into one (primary = first element = 3).
%
% Usage (multiple independent groups):
%   merge_clusters(channels, {[3 5], [1 2 4]}, 'folder', '')
%       Independently merges 3+5 into one cluster and 1+2+4 into another.
%       Each group's primary (target) is its first element.
%
% After renumbering, merge_map records every group so the report can
% annotate each resulting cluster with the former IDs that fed into it.

    p = inputParser;
    addRequired(p, 'channels', @isnumeric);
    addRequired(p, 'merge_list', @(x) isnumeric(x) || iscell(x));
    addParameter(p, 'folder', '', @ischar);
    parse(p, channels, merge_list, varargin{:});

    % Normalise merge_list -> cell array of row vectors, one per group
    if isnumeric(merge_list)
        % Legacy: single numeric vector
        groups = { unique(merge_list(:))' };
    else
        groups = merge_list(:)';
        for g = 1:numel(groups)
            groups{g} = unique(groups{g}(:))';   % ensure row, sorted, no dups
            if numel(groups{g}) < 2
                error('Each merge group must contain at least 2 cluster IDs (group %d).', g);
            end
        end
    end

    root = resolve_session_root();
    load(fullfile(root, 'NSx.mat'), 'NSx');

    if ~isempty(p.Results.folder)
        times_dir = p.Results.folder;
    else
        [~, cur_dir] = fileparts(pwd);
        if startsWith(cur_dir, 'times')
            times_dir = pwd;
        else
            d = dir(fullfile(root, 'times*'));
            d = d([d.isdir]);
            if isempty(d), error('No times directory found.'); end
            [~, idx] = max([d.datenum]);
            times_dir = fullfile(root, d(idx).name);
        end
    end

    results = cell(size(channels));

    for c = 1:length(channels)
        chan = channels(c);

        posch = find(arrayfun(@(x) (x.chan_ID == chan), NSx), 1);
        if isempty(posch)
            warning('Channel %d not found in NSx.mat mapping.', chan);
            continue;
        end

        filename    = sprintf('times_%s.mat', NSx(posch).output_name);
        matched_file = fullfile(times_dir, filename);

        if ~exist(matched_file, 'file')
            warning('File %s does not exist in %s', filename, times_dir);
            continue;
        end

        data = load(matched_file);
        if ~isfield(data, 'cluster_class')
            warning('File %s is missing cluster_class.', filename);
            continue;
        end

        cluster_ids   = data.cluster_class(:, 1);
        unique_before = unique(cluster_ids);

        % Validate all requested IDs exist
        all_requested = unique([groups{:}]);
        if ~all(ismember(all_requested, unique_before))
            missing = all_requested(~ismember(all_requested, unique_before));
            warning('Clusters [%s] not found in channel %d – skipping.', ...
                    num2str(missing), chan);
            continue;
        end

        % --- Apply each group's merge sequentially ---
        % Each group: collapse all members onto the first (target) ID.
        for g = 1:numel(groups)
            grp    = groups{g};
            target = grp(1);
            for s = grp(2:end)
                cluster_ids(cluster_ids == s) = target;
            end
        end

        % --- Contiguous 0-indexed renumbering ---
        new_unique  = unique(cluster_ids);
        renumbered  = zeros(size(cluster_ids));
        for i = 1:length(new_unique)
            renumbered(cluster_ids == new_unique(i)) = i - 1;
        end
        data.cluster_class(:, 1) = renumbered;

        % --- Build former->new remapping (covers all IDs, all groups) ---
        former_arr = unique_before(:);
        new_arr    = zeros(size(former_arr));
        % Build a lookup: former ID -> post-merge (pre-renumber) ID
        post_merge_id = double(unique_before);   % default: identity
        for g = 1:numel(groups)
            grp    = groups{g};
            target = grp(1);
            for s = grp(2:end)
                post_merge_id(unique_before == s) = target;
            end
        end
        for i = 1:numel(former_arr)
            new_arr(i) = find(new_unique == post_merge_id(i), 1) - 1;
        end

        % --- Build per-group tracking (new_id = renumbered target) ---
        %   merge_groups(g).former_ids  – original IDs merged (sorted)
        %   merge_groups(g).new_id      – renumbered result ID
        merge_groups_out = struct('former_ids', {}, 'new_id', {});
        for g = 1:numel(groups)
            grp    = groups{g};
            target = grp(1);
            new_id = find(new_unique == target, 1) - 1;
            merge_groups_out(g).former_ids = grp;   % all members incl. target
            merge_groups_out(g).new_id     = new_id;
        end

        % --- Save pipeline tracking ---
        data.merged    = true;

        % merge_map: flat lookup used by report for quick annotation
        data.merge_map         = struct();
        data.merge_map.former  = former_arr;
        data.merge_map.new     = new_arr;

        % merge_groups: richer per-group record (supports multiple merges)
        data.merge_groups = merge_groups_out;

        % Legacy scalar fields (valid only when exactly one group was merged)
        if numel(groups) == 1
            data.merged_clusters = groups{1};
            data.merged_into     = merge_groups_out(1).new_id;
        else
            % Remove legacy fields that would be misleading for multi-group
            if isfield(data, 'merged_clusters'), data = rmfield(data, 'merged_clusters'); end
            if isfield(data, 'merged_into'),     data = rmfield(data, 'merged_into');     end
        end

        % Historical logging (one entry per group)
        if ~isfield(data, 'par') || ~isstruct(data.par), data.par = struct(); end
        for g = 1:numel(groups)
            history_event = struct( ...
                'timestamp',       datestr(now, 'yyyy-mm-dd HH:MM:SS'), ...
                'merged_clusters', groups{g}, ...
                'into_id',         merge_groups_out(g).new_id ...
            );
            if isfield(data.par, 'merge_history')
                data.par.merge_history(end+1) = history_event;
            else
                data.par.merge_history = history_event;
            end
        end

        % --- Folder name: encode all groups ---
        % e.g. ch[1]_merge[3_5]-[1_2_4]
        group_strs = cellfun(@(g) strrep(mat2str(g),' ','_'), groups, 'UniformOutput', false);
        fold_merge = sprintf('ch%s_merge%s', mat2str(channels), strjoin(group_strs, '-'));
        merged_dir = fullfile(times_dir, fold_merge);
        if ~exist(merged_dir, 'dir'), mkdir(merged_dir); end
        [~, name, ext] = fileparts(filename);
        out_file = fullfile(merged_dir, [name ext]);

        save(out_file, '-struct', 'data');
        results{c} = data;
    end

    if nargout > 0
        if length(channels) == 1
            varargout{1} = results{1};
        else
            varargout{1} = results;
        end
    end
end