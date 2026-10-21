function masked_dir = build_masked_arm(root, ch, exp_type, varargin)
% BUILD_MASKED_ARM  Just the masked arm -- nothing else. Copies the newest
% spikes* folder, runs within_channel + refract_viol on that copy,
% Do_features + Do_clustering into a fresh times folder, snapshots
% spikes+times into its own folder, builds THAT arm's own grapes_blanks.mat.
%
% Mirrors validate_recluster_vs_handselect.m's masked_recluster arm step by
% step (same cd(root), same copy/within_channel/refract_viol/Do_features/
% Do_clustering/snapshot/do_structure_mu+_sorted sequence) -- but ONLY that,
% none of the handselect reconstruction, characterize_grapes_responses,
% sorting metrics, class matching, or report that tool also does. That tool
% is not called here at all.
%
% USAGE
%   masked_dir = build_masked_arm(root, ch, exp_type)
%
%   Name/value:
%     'spikes_src'                 ''      spikes* folder to copy from
%                                           (default: newest under root)
%     'within_channel_detections'  'all'
%     'refract_keep_strategy'      'none'
%     'seed'                       0       production default (clock-seeded).
%                                           Pinning this only matters when
%                                           comparing multiple reclustering
%                                           runs against each other; here
%                                           there's exactly one Do_clustering
%                                           call, so there's nothing to keep
%                                           reproducible against.
%     'clu_par'                    struct()  overrides on top of the default
%                                           (byte-identical to processing_
%                                           steps_MCW_reduced's) clustering par
%     'circshiftblanks'            false
%     'out_dir'                    ''      default: <root>/masked_arm_<ts>
%
% OUTPUT
%   masked_dir   folder with <nm>_spikes.mat, times_<nm>.mat,
%                templates_wc_offline.mat, grapes_blanks.mat

    p = inputParser;
    addParameter(p, 'spikes_src', '');
    addParameter(p, 'within_channel_detections', 'all');
    addParameter(p, 'refract_keep_strategy', 'none');
    addParameter(p, 'seed', 0);
    addParameter(p, 'clu_par', struct());
    addParameter(p, 'circshiftblanks', false);
    addParameter(p, 'out_dir', '');
    parse(p, varargin{:});
    o = p.Results;

    addpath(genpath('/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline'));
    cd(root);

    load(fullfile(root, 'NSx.mat'), 'NSx');
    nm = output_name_for_local(NSx, ch);
    assert(~isempty(nm), 'channel %d not found in NSx.mat', ch);

    ts = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));

    % ---- working spikes copy (same rule within_channel/refract_viol/
    % Do_clustering use: newest spikes* under pwd) --------------------------
    src = o.spikes_src;
    if isempty(src)
        d = dir(fullfile(root, 'spikes*')); d = d([d.isdir]);
        d = d(~startsWith({d.name}, {'spikes_recval_', 'spikes_masked_'}));
        assert(~isempty(d), 'no spikes* folder under %s; pass ''spikes_src''.', root);
        [~, i] = max([d.datenum]);
        src = fullfile(root, d(i).name);
    elseif ~isfolder(src)
        src = fullfile(root, src);
    end
    spk_work = fullfile(root, ['spikes_masked_' ts]);
    mkdir(spk_work);
    copy_if_local(fullfile(src, [nm '_spikes.mat']), spk_work);
    assert(exist(fullfile(spk_work, [nm '_spikes.mat']), 'file') == 2, ...
        'no %s_spikes.mat in %s', nm, src);
    fprintf('[build_masked_arm] working spikes: %s (from %s)\n', spk_work, src);

    % ---- mask (within_channel + refract_viol operate on the newest spikes*
    % under pwd, which is spk_work since it was just created) ---------------
    within_channel(ch, 'detections', o.within_channel_detections,'parallel',false);
    refract_viol(ch, 'keep_strategy', o.refract_keep_strategy,'parallel',false);

    % ---- cluster (byte-identical default clustering par to
    % processing_steps_MCW_reduced, unless overridden) -----------------------
    tim_dir = fullfile(root, ['times_masked_' ts]);
    mkdir(tim_dir);
    fprintf('[build_masked_arm] Do_features\n');
    Do_features(ch, 'parallel', false, 'folder', spk_work);
    clu_par = struct('min_clus', 20, 'max_spk', 30000, 'mintemp', 0, 'maxtemp', 0.251, ...
                     'tempstep', 0.01, 'max_std_templates', 3, 'max_spikes_plot', 5000);
    for f = fieldnames(o.clu_par)', clu_par.(f{1}) = o.clu_par.(f{1}); end
    fprintf('[build_masked_arm] Do_clustering (seed=%g, maxtemp=%g, min_clus=%g)\n', ...
            o.seed, clu_par.maxtemp, clu_par.min_clus);
    Do_clustering(ch, 'parallel', false, 'make_times', true, 'make_templates', true, ...
                  'make_plots', false, 'par', clu_par, 'seed', o.seed, ...
                  'folder', spk_work, 'times_folder', tim_dir);
    try
        temperature_diagram(tim_dir, nm, clu_par, fullfile(tim_dir, ['temperature_' nm '.png']));
    catch ME
        warning('%s', ['[build_masked_arm] temperature_diagram failed: ' ME.message]);
    end

    % ---- snapshot spikes+times(+templates) into the arm's own folder ------
    if isempty(o.out_dir)
        masked_dir = fullfile(root, ['masked_arm_' ts]);
    else
        masked_dir = o.out_dir;
    end
    if ~exist(masked_dir, 'dir'), mkdir(masked_dir); end
    copy_if_local(fullfile(spk_work, [nm '_spikes.mat']), masked_dir);
    copy_if_local(fullfile(tim_dir, ['times_' nm '.mat']), masked_dir);
    copy_if_local(fullfile(tim_dir, 'templates_wc_offline.mat'), masked_dir);
    copy_if_local(fullfile(tim_dir, ['temperature_' nm '.png']), masked_dir);

    % ---- this arm's own grapes_blanks.mat (mu + classN, no rescue) --------
    do_structure_mu_BCM_online3(ch, exp_type, true, o.circshiftblanks, false, 'spike_dir', masked_dir);
    do_structure_sorted_BCM_online3(ch, true, o.circshiftblanks, false, ...
                                    'times_dir', masked_dir, 'spike_dir', masked_dir);

    fprintf('[build_masked_arm] done -> %s\n', masked_dir);
end

function nm = output_name_for_local(NSx, ch)
    pos = find(arrayfun(@(x) x.chan_ID == ch, NSx), 1);
    if isempty(pos), nm = ''; else, nm = NSx(pos).output_name; end
end

function copy_if_local(src, dest_dir)
    if exist(src, 'file') == 2, copyfile(src, dest_dir); end
end

function [nkeep, elbow_row, sel_tc] = find_temp_curve(tree, clu, par)
% Port of clustering/find_temp.m that also returns nkeep = the CUMULATIVE count
% of clusters find_temp has committed to by each temperature (so it rises to the
% final class count and stays flat), the elbow row, and the finally-selected
% [temp_row, cluster_rank] pairs (sel_tc). Mirrors that file line-for-line so
% the panel matches the pipeline; the overlap-removal pass is skipped when the
% .dg_01.lab labels (clu) are unavailable.
    min_clus = par.min_clus; c_ov = par.c_ov; elbow_min = par.elbow_min;
    y = tree(1:end, 5);
    maxdiff = max(diff(tree(1:end, 6:end)), [], 2); maxdiff(maxdiff < 0) = 0;
    prop = (y(2:end) + maxdiff .* (maxdiff > 0)) ./ y(1:end-1);
    aux = find(prop < elbow_min, 1, 'first') + 1;
    if ~isempty(aux) && par.mintemp == 0 && aux == 2
        aux = find(prop(2:end) < elbow_min, 1, 'first') + 2;
    end
    tr = tree(1:end-1, 5:end);
    clus = zeros(size(tr)); clus(tr >= min_clus) = 1;
    dt = diff(tr);
    clus = clus & [ones(1, size(clus, 2)); dt > min_clus];
    for ii = 1:size(clus, 1)
        d = find(clus(ii, :), 1, 'last');
        if ~isempty(d), clus(ii, 1:d) = 1; end
    end
    elbow_row = size(tr, 1);
    if ~isempty(aux), clus(aux:end, :) = 0; elbow_row = aux; end
    if ~isempty(clu) && size(clu, 1) >= size(tr, 1) + 1
        cl = clu(1:end-1, 3:end) + 1;
        for ti = size(clus, 1):-1:1
            detect = find(clus(ti, :));
            for ci = 1:numel(detect)
                a = (cl(ti, :) == detect(ci));
                for tj = ti-1:-1:1
                    rem = find(clus(tj, :));
                    for j = 1:numel(rem)
                        b = (cl(tj, :) == rem(j));
                        if nnz(a & b) / max(min(nnz(b), nnz(a)), 1) >= c_ov
                            clus(tj, rem(j)) = 0;
                        end
                    end
                end
            end
        end
    end
    % cumulative: "# clusters find_temp has committed to by temperature T"
    nkeep = [cumsum(sum(clus, 2)); NaN];
    nkeep = nkeep(1:size(tree, 1));
    [tt, cc] = find(clus);
    sel_tc = sortrows([tt, cc]);
end

function save_fig_local(f, out_png)
    set(f, 'Color', 'w');
    try
        print(f, out_png, '-dpng', '-r130');
    catch
        exportgraphics(f, out_png, 'Resolution', 130, 'BackgroundColor', 'white');
    end
    close(f);
end
