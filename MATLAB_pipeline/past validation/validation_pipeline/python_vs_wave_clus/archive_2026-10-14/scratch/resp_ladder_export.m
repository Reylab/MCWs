function resp_ladder_export(base)
% base/<ch>/<sort>/resp_<unit>_ch<ch>.mat -> base/ladder_responses.csv (unit, sort, class, stim metrics)
F = dir(fullfile(base, '*', '*', 'resp_*.mat'));
rows = {};
for f = F'
    o = load(fullfile(f.folder, f.name));
    tok = regexp(f.name, '^resp_(.+)_ch\d+\.mat$', 'tokens', 'once');
    [~, sort_] = fileparts(f.folder);
    for k = fieldnames(o)'
        x = o.(k{1});
        if ~isstruct(x) || ~isfield(x, 'metrics'), continue; end
        T = x.metrics; v = T.Properties.VariableNames;
        ok = cellfun(@(c) isnumeric(T.(c)) || islogical(T.(c)) || isstring(T.(c)) || (iscell(T.(c)) && all(cellfun(@ischar, T.(c)))), v);
        T = T(:, v(ok));
        T.unit = repmat(string(tok{1}), height(T), 1); T.sort = repmat(string(sort_), height(T), 1); T.grapes = repmat(string(k{1}), height(T), 1);
        rows{end+1} = T; %#ok<AGROW>
    end
end
common = rows{1}.Properties.VariableNames;
for i = 2:numel(rows), common = intersect(common, rows{i}.Properties.VariableNames, 'stable'); end
rows = cellfun(@(t) t(:, common), rows, 'UniformOutput', false);
writetable(vertcat(rows{:}), fullfile(base, 'ladder_responses.csv'));
fprintf('%d files -> %s\n', numel(F), fullfile(base, 'ladder_responses.csv'));
end
