function T = exclude_spike_times(T, cfg, label)
% EXCLUDE_SPIKE_TIMES  Drop the spikes of a loaded times file that lie within cfg.tol ms of any time
% in cfg.exclude_ms (e.g. GT's bundle collisions, gt_collision_times). Every per-spike field is cut
% the same way (cluster_class, spikes, index, forced, inspk / features, rescue_mask, quarantine.*,
% ...), so the rows stay aligned. Only the loaded copy changes -- never the file.
% No cfg.exclude_ms (or empty): T is returned unchanged.
if ~isfield(cfg, 'exclude_ms') || isempty(cfg.exclude_ms) || ~isfield(T, 'cluster_class'), return; end
if nargin < 3, label = ''; end
t = T.cluster_class(:, 2); n = numel(t);
ex = sort(cfg.exclude_ms(:));
j = discretize(t, [-Inf; ex; Inf]);
lo = max(j - 1, 1); hi = min(j, numel(ex));
drop = abs(t - ex(lo)) <= cfg.tol | abs(t - ex(hi)) <= cfg.tol;
if ~any(drop), return; end
keep = ~drop;
skip = {'par', 'gui_status', 'clustering_metadata', 'template_match', 'ipermut', 'Temp', 'source_note'};
for f = fieldnames(T)'
    if ismember(f{1}, skip), continue; end
    T.(f{1}) = cut(T.(f{1}), keep, n);
end
if ~isempty(label)
    fprintf('[exclude_spike_times] %s: left out %d of %d spikes (within %.2f ms of an excluded time)\n', label, nnz(drop), n, cfg.tol);
end
end

function v = cut(v, keep, n)
if isstruct(v) && isscalar(v)
    for f = fieldnames(v)', v.(f{1}) = cut(v.(f{1}), keep, n); end
elseif (isnumeric(v) || islogical(v)) && ~isempty(v)
    if size(v, 1) == n
        v = v(keep, :);
    elseif isvector(v) && numel(v) == n
        v = v(keep);
    end
end
end
