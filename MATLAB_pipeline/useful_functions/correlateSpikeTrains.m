function results = correlateSpikeTrains(data1, data2, tolerance, varargin)
% correlateSpikeTrains: Correlates two sets of [ClusterID, TimeMS]
% Inputs:
%   data1     - [ClusterID, TimeMS] (N x 2)  — t1, typically MATLAB
%   data2     - [ClusterID, TimeMS] (M x 2)  — t2, typically Python
%   tolerance - Match window in ms (e.g., 0.025)
%
% Optional name-value:
%   'label1'  - string label for dataset 1 (default: 'Dataset 1 (MATLAB)')
%   'label2'  - string label for dataset 2 (default: 'Dataset 2 (Python)')
%   'par1'    - par struct from dataset 1 spikes file (enables filter comparison)
%   'par2'    - par struct from dataset 2 spikes file (enables filter comparison)
%
% Output results struct fields:
%   .shared_overlap  - TABLE: idx1,idx2,clust1,clust2,t1,t2,dt
%   .unique_data1    - TABLE: orig_idx,clust,t
%   .unique_data2    - TABLE: orig_idx,clust,t
%   .filter_match    - true if filter parameters are identical across datasets

    p = inputParser;
    addParameter(p, 'label1', 'Dataset 1 (MATLAB)', @ischar);
    addParameter(p, 'label2', 'Dataset 2 (Python)',  @ischar);
    addParameter(p, 'par1',   [],  @(x) isstruct(x) || isempty(x));
    addParameter(p, 'par2',   [],  @(x) isstruct(x) || isempty(x));
    parse(p, varargin{:});
    label1 = p.Results.label1;
    label2 = p.Results.label2;
    par1   = p.Results.par1;
    par2   = p.Results.par2;

    results = struct();

    % ── 1. Nearest Neighbour Search ───────────────────────────────────────
    [idx2_in_1, dist12] = knnsearch(data2(:,2), data1(:,2));
    [idx1_in_2, dist21] = knnsearch(data1(:,2), data2(:,2));

    mask1 = dist12 <= tolerance;
    mask2 = dist21 <= tolerance;

    % ── 2. Shared overlap table ───────────────────────────────────────────
    shared_idx1  = find(mask1);
    shared_idx2  = idx2_in_1(mask1);
    shared_clus1 = data1(shared_idx1, 1);
    shared_clus2 = data2(shared_idx2, 1);
    shared_t1    = data1(shared_idx1, 2);
    shared_t2    = data2(shared_idx2, 2);
    shared_dt    = (shared_t1 - shared_t2) / 1000;   % ms → s

    results.shared_overlap = table(...
        shared_idx1, shared_idx2, shared_clus1, shared_clus2, ...
        shared_t1, shared_t2, shared_dt, ...
        'VariableNames', {'idx1','idx2','clust1','clust2','t1','t2','dt'});

    % ── 3. Unique spike tables ────────────────────────────────────────────
    uniq1_orig  = find(~mask1);
    results.unique_data1 = table(uniq1_orig, data1(~mask1,1), data1(~mask1,2), ...
        'VariableNames', {'orig_idx','clust','t'});

    uniq2_orig  = find(~mask2);
    results.unique_data2 = table(uniq2_orig, data2(~mask2,1), data2(~mask2,2), ...
        'VariableNames', {'orig_idx','clust','t'});

    n_shared  = height(results.shared_overlap);
    n_unique1 = height(results.unique_data1);
    n_unique2 = height(results.unique_data2);

    % ── 4. Filter parameter comparison ───────────────────────────────────
    results.filter_match = true;
    if ~isempty(par1) && ~isempty(par2)
        fields_to_check = {'sort_order','sort_fmin','sort_fmax',...
                           'detect_order','detect_fmin','detect_fmax',...
                           'stdmin','stdmax','ref_ms','w_pre','w_post',...
                           'interpolation','sr'};
        fprintf('\n=== Filter / Parameter Comparison ===\n');
        fprintf('%-20s  %-20s  %-20s  %s\n','Parameter','MATLAB','Python','Match?');
        fprintf('%s\n', repmat('-',1,72));
        for k = 1:length(fields_to_check)
            f = fields_to_check{k};
            has1 = isfield(par1,f);
            has2 = isfield(par2,f);
            if ~has1 && ~has2; continue; end
            v1 = '(missing)'; v2 = '(missing)'; match_str = '';
            if has1; v1 = format_val(par1.(f)); end
            if has2; v2 = format_val(par2.(f)); end
            if has1 && has2
                if isnumeric(par1.(f)) && isnumeric(par2.(f))
                    ok = isequal(size(par1.(f)),size(par2.(f))) && ...
                         all(abs(par1.(f)(:) - par2.(f)(:)) < 1e-9);
                elseif ischar(par1.(f)) && ischar(par2.(f))
                    ok = strcmp(par1.(f), par2.(f));
                else
                    ok = false;
                end
                if ok; match_str = 'OK';
                else;  match_str = '*** MISMATCH ***';
                       results.filter_match = false;
                end
            end
            fprintf('%-20s  %-20s  %-20s  %s\n', f, v1, v2, match_str);
        end
        fprintf('%s\n', repmat('-',1,72));
        if results.filter_match
            fprintf('All checked parameters match.\n');
        else
            fprintf('*** Parameter mismatches detected — these affect detection.\n');
        end
    end

    % ── 5. Printed report ─────────────────────────────────────────────────
    fprintf('\n==============================================\n');
    fprintf('         SPIKE CORRELATION REPORT\n');
    fprintf('==============================================\n');
    fprintf('Tolerance: %.4f ms\n', tolerance);
    fprintf('%-22s | %-10s | %-10s\n', 'Dataset', 'Shared', 'Unique');
    fprintf('----------------------------------------------\n');
    fprintf('%-22s | %-10d | %-10d\n', label1, n_shared, n_unique1);
    fprintf('%-22s | %-10d | %-10d\n', label2, n_shared, n_unique2);
    fprintf('==============================================\n');

    totalSpikes1 = size(data1,1);
    matchRate    = (n_shared / totalSpikes1) * 100;
    fprintf('Global Match Rate (%s): %.2f%%\n', label1, matchRate);
    fprintf('Count difference: %d spikes\n', abs(n_unique1 - n_unique2));

    % ── 6. Unique spike timing — are they clustered or spread? ───────────
    fprintf('\nUnique to %s (%d spikes):\n', label1, n_unique1);
    if n_unique1 > 0
        u1t = results.unique_data1.t;
        fprintf('  Time range: %.2f – %.2f ms\n', min(u1t), max(u1t));
        fprintf('  First 10 times (ms): ');
        fprintf('%.3f ', u1t(1:min(10,end))); fprintf('\n');
    end
    fprintf('Unique to %s (%d spikes):\n', label2, n_unique2);
    if n_unique2 > 0
        u2t = results.unique_data2.t;
        fprintf('  Time range: %.2f – %.2f ms\n', min(u2t), max(u2t));
        fprintf('  First 10 times (ms): ');
        fprintf('%.3f ', u2t(1:min(10,end))); fprintf('\n');
    end

    % ── 7. Timing jitter stats on shared spikes ───────────────────────────
    if n_shared > 0
        jitter = results.shared_overlap.t1 - results.shared_overlap.t2;
        fprintf('\nShared spike timing jitter (t1-t2, ms):\n');
        fprintf('  Mean=%.5f  Median=%.5f  Std=%.5f  Max|jitter|=%.5f\n',...
            mean(jitter), median(jitter), std(jitter), max(abs(jitter)));
    end

    % ── 8. Cluster mapping ────────────────────────────────────────────────
    if n_shared > 0
        [results.overlapMatrix, ~, results.clusterLabels] = ...
            crosstab(results.shared_overlap.clust1, results.shared_overlap.clust2);

        figure('Color','w','Name','Cluster Mapping Analysis');
        subplot(1,2,1);
        heatmap(results.overlapMatrix, 'Title','Spike Match Counts',...
            'XLabel',[label2 ' Clusters'],'YLabel',[label1 ' Clusters']);
        subplot(1,2,2);
        prob = results.overlapMatrix ./ sum(results.overlapMatrix,2);
        heatmap(prob,'Title','Mapping Probability',...
            'XLabel',[label2 ' Clusters'],'YLabel',[label1 ' Clusters']);
        colormap(parula);
    end

    % ── 9. Jitter + correlogram plots ────────────────────────────────────
    if n_shared > 0
        jitter = results.shared_overlap.t1 - results.shared_overlap.t2;
        figure('Color','w','Name','Timing Jitter');
        subplot(1,2,1);
        histogram(jitter, 50, 'FaceColor',[0.4 0.6 0.8]);
        title('Global Timing Jitter (All Spikes)');
        xlabel('Time Difference (ms)'); ylabel('Count');
        xline(0,'r--');

        subplot(1,2,2);
        [counts, edges] = histcounts(jitter, 100);
        centers = (edges(1:end-1)+edges(2:end))/2;
        stem(centers, counts, 'Marker','none');
        title('Global Cross-Correlogram');
        xlabel('Lag (ms)');
    end

    % ── 10. Unique spike timing distribution ─────────────────────────────
    if n_unique1 > 0 || n_unique2 > 0
        figure('Color','w','Name','Unique Spike Timing');
        subplot(2,1,1);
        if n_unique1 > 0
            histogram(results.unique_data1.t, 50, 'FaceColor','b','EdgeColor','none');
        end
        title(sprintf('Unique to %s (%d spikes) — timing distribution', label1, n_unique1));
        xlabel('Time (ms)'); ylabel('Count');

        subplot(2,1,2);
        if n_unique2 > 0
            histogram(results.unique_data2.t, 50, 'FaceColor','r','EdgeColor','none');
        end
        title(sprintf('Unique to %s (%d spikes) — timing distribution', label2, n_unique2));
        xlabel('Time (ms)'); ylabel('Count');
        sgtitle('Are unique spikes clustered (edge artifact?) or spread (threshold difference)?');
    end

    % Add to the bottom of correlateSpikeTrains.m
    chunk_ms = 5 * 60 * 1000; % Assuming 5 minute chunks
    if n_unique1 > 0 || n_unique2 > 0
        figure('Color','w','Name','Unique Spikes vs Chunk Boundary');
        if n_unique1 > 0
            scatter(mod(results.unique_data1.t, chunk_ms), ones(n_unique1,1), 'b.'); hold on;
        end
        if n_unique2 > 0
            scatter(mod(results.unique_data2.t, chunk_ms), ones(n_unique2,1)*1.1, 'r.');
        end
        title('Unique Spikes within a 5-Minute Chunk Cycle');
        xlabel('Time since chunk start (ms)');
        ylim([0.8 1.3]); yticks([1 1.1]); yticklabels({label1, label2});
    end
end

function s = format_val(v)
    if ischar(v);      s = v;
    elseif isscalar(v); s = num2str(v);
    else;              s = mat2str(v,4);
    end
end