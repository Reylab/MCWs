function ST = collect_sorting_metrics(trials, channels, cmp_dir, varargin)
% COLLECT_SORTING_METRICS  Stack every trial's compute_metrics_batch output
% into one long table (trial, channel, cluster_id, <metric cols>), save
% csv/mat, and draw a per-channel plot of each metric across trials (one
% line per cluster). Shared by validate_mask_effects.m (all clusters, no
% filter) and analyze_cluster_membership.m (filtered to the one matched
% cluster per trial, via 'cluster_filter', so "line per cluster" becomes
% one line = the tracked class across trials with different arm_class ids).
%
% REQUIRES trials(t).dir (folder with 'metrics/' or 'metrics rescue/',
% i.e. compute_metrics_batch's own output -- run it first) and
% trials(t).rescue (logical, picks the rescue metrics subfolder).
%
% Name/value:
%   'cluster_filter'  containers.Map, trial name -> cluster_id to keep for
%                      that trial (default: empty map = keep every cluster,
%                      original behaviour). Kept rows get their cluster_id
%                      relabelled to a fixed sentinel (0) so they plot as
%                      ONE line even though the matched arm_class differs
%                      per trial; the true id is kept in 'matched_cluster_id'.
%   'file_prefix'      base name for the csv/mat/png (default 'sorting_metrics')
%   'title_suffix'     appended to the plot title (default '')

    p = inputParser;
    addParameter(p, 'cluster_filter', containers.Map('KeyType','char','ValueType','double'));
    addParameter(p, 'file_prefix', 'sorting_metrics', @ischar);
    addParameter(p, 'title_suffix', '', @ischar);
    parse(p, varargin{:});
    o = p.Results;

    ST = table();
    for t = 1:numel(trials)
        mdir = fullfile(trials(t).dir, 'metrics');
        if trials(t).rescue && exist(fullfile(trials(t).dir,'metrics rescue'),'dir')
            mdir = fullfile(trials(t).dir, 'metrics rescue');
        end
        d = dir(fullfile(mdir, '*_metrics.mat'));
        for k = 1:numel(d)
            S = load(fullfile(d(k).folder, d(k).name));
            if ~isfield(S,'metrics_table') || ~istable(S.metrics_table) || isempty(S.metrics_table)
                continue;
            end
            Tt = S.metrics_table;
            Tt.trial = repmat(string(trials(t).name), height(Tt), 1);
            if isKey(o.cluster_filter, trials(t).name)
                cid = o.cluster_filter(trials(t).name);
                Tt = Tt(Tt.cluster_id == cid, :);
                if isempty(Tt), continue; end
                Tt.matched_cluster_id = Tt.cluster_id;
                Tt.cluster_id = zeros(height(Tt), 1);   % sentinel -> one line across trials
            end
            ST = [ST; Tt];
        end
    end
    if isempty(ST), return; end

    % move trial / channel_id to the front
    lead = intersect({'trial','channel_id','cluster_id','num_spikes'}, ST.Properties.VariableNames, 'stable');
    ST = movevars(ST, lead, 'Before', 1);
    try
        writetable(ST, fullfile(cmp_dir, [o.file_prefix '.csv']));
    catch
    end
    save(fullfile(cmp_dir, [o.file_prefix '.mat']), 'ST');

    exclude_cols = {'trial','channel_id','cluster_id','filename','matched_cluster_id'};
    metric_cols = setdiff(ST.Properties.VariableNames, exclude_cols, 'stable');
    metric_cols = metric_cols(varfun(@(c) isnumeric(c), ST(:,metric_cols), ...
                                     'OutputFormat','uniform'));
    tnames = string({trials.name});
    for ci = 1:numel(channels)
        ch = channels(ci);
        sub = ST(ST.channel_id == ch, :);
        if isempty(sub), continue; end
        cids = unique(sub.cluster_id).';
        f = figure('Visible','off','Position',[100 100 max(1100,260*numel(tnames)) 780],'Color','w');
        tl = tiledlayout(f,'flow','Padding','loose','TileSpacing','loose');
        title(tl, sprintf('ch%d  sorting metrics across trials  (line per cluster)%s', ch, o.title_suffix), ...
              'Interpreter','none');
        xlabel(tl, strjoin(compose('%d=%s',(1:numel(tnames))',tnames(:)),'   '), 'Interpreter','none');
        for mm = 1:numel(metric_cols)
            ax = nexttile(tl); hold(ax,'on');
            for cc = cids
                y = nan(1,numel(tnames));
                for tt = 1:numel(tnames)
                    r = sub(sub.cluster_id==cc & sub.trial==tnames(tt), :);
                    if ~isempty(r), y(tt) = double(r.(metric_cols{mm})(1)); end
                end
                plot(ax, 1:numel(tnames), y, '-o', 'DisplayName', sprintf('c%d',cc));
            end
            hold(ax,'off'); grid(ax,'on');
            xlim(ax,[0.5 numel(tnames)+0.5]); xticks(ax,1:numel(tnames));
            title(ax, metric_cols{mm}, 'Interpreter','none');
        end
        exportgraphics(f, fullfile(cmp_dir, sprintf('%s_ch%d.png', o.file_prefix, ch)), ...
                       'Resolution', 130, 'BackgroundColor','white');
        close(f);
    end
    fprintf('[collect_sorting_metrics] -> %s (%s.csv + per-channel PNGs)\n', cmp_dir, o.file_prefix);
end
