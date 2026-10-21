SC = '/mnt/data0/MCWs Sessions/matlab validation/single channel';
D = dir(SC); D = D([D.isdir] & ~startsWith({D.name}, '.'));
rows = {};
for d = D'
    root = fullfile(SC, d.name);
    loc = {fullfile(root, 'hand selected'), root};
    for L = loc
        t = dir(fullfile(L{1}, 'times_*.mat'));
        if isempty(t), continue; end
        tok = regexp(t(1).name, 'raw_(\d+)', 'tokens', 'once'); ch = str2double(tok{1});
        T = load(fullfile(L{1}, t(1).name), 'cluster_class');
        nt = arrayfun(@(c) nnz(T.cluster_class(:,1) == c), 1:max(T.cluster_class(:,1)));
        g = [dir(fullfile(L{1}, 'grapes*.mat')); dir(fullfile(L{1}, 'Copy_of_grapes*.mat'))];
        g = g(~contains({g.name}, 'response_metrics'));
        if isempty(g)
            rows(end+1, :) = {d.name, strrep(L{1}, [root filesep], ''), t(1).name, mat2str(nt), '(no grapes file)', ''}; %#ok<AGROW>
        end
        for gi = 1:numel(g)
            try
                R = load(fullfile(L{1}, g(gi).name), 'rasters');
                dd = R.rasters.(sprintf('chan%d', ch)).details;
                f = fieldnames(dd); f = f(~cellfun(@isempty, regexp(f, '^class\d+$', 'once')));
                ng = zeros(1, numel(f));
                for k = 1:numel(f), ng(str2double(f{k}(6:end))) = dd.(f{k}); end
                if isequal(nt, ng), st = 'MATCH'; else, st = 'MISMATCH'; end
                rows(end+1, :) = {d.name, strrep(L{1}, [root filesep], ''), t(1).name, mat2str(nt), sprintf('%s %s', g(gi).name, mat2str(ng)), st}; %#ok<AGROW>
            catch ME
                rows(end+1, :) = {d.name, strrep(L{1}, [root filesep], ''), t(1).name, mat2str(nt), sprintf('%s (unreadable: %s)', g(gi).name, ME.message), 'ERR'}; %#ok<AGROW>
            end
        end
        break   % hand selected wins when present
    end
end
for i = 1:size(rows, 1)
    fprintf('%-9s | %-62s | %-14s | times %s\n          | grapes: %s\n', rows{i,6}, rows{i,1}, rows{i,2}, rows{i,4}, rows{i,5});
end
