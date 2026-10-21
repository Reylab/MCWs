function resp_export(base)
% all grapes_response_metrics_*.mat under base/<ch>/<sort>/ -> base/all_metrics.csv
F = dir(fullfile(base, '*', '*', 'grapes_response_metrics_*.mat'));
rows = {};
for f = F'
    o = load(fullfile(f.folder, f.name));
    tg = fieldnames(o);
    for k = 1:numel(tg)
        x = o.(tg{k});
        if ~isstruct(x) || ~isfield(x, 'metrics'), continue; end
        T = x.metrics;
        [~, sort_] = fileparts(f.folder); [~, chd] = fileparts(fileparts(f.folder));
        v = T.Properties.VariableNames;
        ok = cellfun(@(c) isnumeric(T.(c)) || islogical(T.(c)) || isstring(T.(c)) || (iscell(T.(c)) && all(cellfun(@ischar, T.(c)))), v);
        T = T(:, v(ok));
        T.sort = repmat(string(sort_), height(T), 1);
        T.chdir = repmat(string(chd), height(T), 1);
        T.grapes = repmat(string(tg{k}), height(T), 1);
        rows{end+1} = T; %#ok<AGROW>
    end
end
common = rows{1}.Properties.VariableNames;
for i = 2:numel(rows), common = intersect(common, rows{i}.Properties.VariableNames, 'stable'); end
rows = cellfun(@(t) t(:, common), rows, 'UniformOutput', false);
writetable(vertcat(rows{:}), fullfile(base, 'all_metrics.csv'));
fprintf('%s\n', strjoin(common, ', '));
end
