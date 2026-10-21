function out = mask_audit(spikes_file, varargin)
% MASK_AUDIT  Per-detection breakdown of within_channel's individual checks
% and refract_viol's keep_strategy outcomes -- straight from fields already
% saved on a *_spikes.mat file. NO re-running within_channel/refract_viol,
% NO clustering: within_channel always computes+saves all 5 detections' fail
% masks in `quarantine_properties` regardless of which were 'active', and
% refract_viol's 3 keep_strategy outcomes are pure functions of ISI gaps in
% `index_all`, re-derivable in one pass without calling it 3 times.
%
% Answers "does any individual mask need loosening?" by (a) how many spikes
% each check flags on its own vs in combination with the others, and (b),
% optionally, how many of a check's flagged spikes actually look like a real
% cluster (shape-close to a template built from a real clustering) -- a
% check that's mostly flagging shape-real spikes is a candidate to loosen.
%
% USAGE
%   out = mask_audit(spikes_file)
%   out = mask_audit(spikes_file, 'times_file', times_file, 'out_dir', d)
%
% INPUTS
%   spikes_file   a *_spikes.mat that has been through within_channel (needs
%                 `quarantine_properties` with fail_structural/amp/width/
%                 prom_ratio/multipeak) and has `index_all`/`spikes_all`.
%
%   Name/value:
%     'out_dir'          ''    default: alongside spikes_file, ./mask_audit/
%     'ref_ms'           []    refractory window (ms) for the refract_viol
%                               re-derivation; default par.ref/(sr/1000), else
%                               1.5 ms (same fallback refract_viol.m uses)
%     'times_file'       ''    OPTIONAL. If given, cross-references each
%                               check's flagged spikes against the REAL
%                               clustering in this times_*.mat (cluster_class,
%                               class>0 only) via nearest-template distance
%                               (raw waveform space, same convention as
%                               force_membership_wc/rescue_spikes) -- reports
%                               what fraction of each check's flagged
%                               population is shape-close to a real unit.
%     'reject_sdnum'     3     "shape-close" radius (in that class's own std)
%                               used only for the times_file cross-reference
%     'rescue_trials'    []    OPTIONAL. struct array with .name/.times_file
%                               (the exact shape build_rescue_sd_trials.m
%                               returns -- pass it straight through). For each
%                               trial, reads `rescue_mask` (already saved by
%                               rescue_spikes.m, aligned to this spikes file's
%                               index_all) and reports, per within_channel
%                               check, what %% of ITS flagged spikes that
%                               particular sdnum actually rescued.
%
% OUTPUT (out struct + files under out_dir):
%   mask_audit_wc_counts.csv/png    per within_channel detection: n_flagged,
%                                    n_unique (flagged by only this check),
%                                    and (if rescue_trials given) pct_rescued
%                                    per sdnum trial -- second panel of the
%                                    same PNG
%   mask_audit_wc_overlap.png       5x5 pairwise Jaccard overlap heatmap
%   mask_audit_refract.csv/png      per keep_strategy: n_removed, n_chains,
%                                    chain-length histogram
%   mask_audit_template.csv/png     (if times_file given) per check: % of its
%                                    flagged spikes shape-close to a real class

    p = inputParser;
    addParameter(p, 'out_dir', '');
    addParameter(p, 'ref_ms', []);
    addParameter(p, 'times_file', '');
    addParameter(p, 'reject_sdnum', 3);
    addParameter(p, 'rescue_trials', []);
    parse(p, varargin{:});
    o = p.Results;

    SPK = load(spikes_file);
    assert(isfield(SPK, 'quarantine_properties'), ...
        'mask_audit: %s has no quarantine_properties -- run within_channel first.', spikes_file);
    qp = SPK.quarantine_properties;
    if isfield(SPK, 'index_all'), index_all = double(SPK.index_all(:));
    else,                          index_all = double(SPK.index(:));  end
    if isfield(SPK, 'spikes_all'), spikes_all = double(SPK.spikes_all);
    else,                          spikes_all = double(SPK.spikes);   end
    n = numel(index_all);

    if isempty(o.out_dir)
        [d0, ~, ~] = fileparts(spikes_file);
        o.out_dir = fullfile(d0, 'mask_audit');
    end
    if ~exist(o.out_dir, 'dir'), mkdir(o.out_dir); end

    % ======================= within_channel per-check ======================
    checks = {'structural', 'amp', 'width', 'prom_ratio', 'multipeak'};
    fld    = strcat('fail_', checks);
    have   = cellfun(@(f) isfield(qp, f), fld);
    checks = checks(have); fld = fld(have);
    nck = numel(checks);
    F = false(n, nck);
    for i = 1:nck, F(:,i) = logical(qp.(fld{i})(:)); end

    n_flagged = sum(F, 1);
    n_unique  = arrayfun(@(i) nnz(F(:,i) & ~any(F(:, [1:i-1, i+1:nck]), 2)), 1:nck);
    T_wc = table(checks(:), n_flagged(:), n_unique(:), 100*n_flagged(:)/n, ...
        'VariableNames', {'detection','n_flagged','n_unique_to_this_check','pct_of_all_spikes'});

    % ---- % of each check's flagged spikes that actually got rescued -------
    % rescue_mask is saved by rescue_spikes.m onto the RESCUE TRIAL's times
    % file, aligned to THIS spikes file's index_all (rescue_sd_sweep trials
    % are built from a copy of this exact spikes file, so the order matches).
    rescue_names = {}; pct_rescued = zeros(nck, 0);
    if ~isempty(o.rescue_trials)
        for ri = 1:numel(o.rescue_trials)
            rt = o.rescue_trials(ri);
            if ~exist(rt.times_file, 'file')
                warning('mask_audit: rescue trial "%s" times_file missing, skipped.', rt.name);
                continue
            end
            RT = load(rt.times_file, 'rescue_mask');
            if ~isfield(RT, 'rescue_mask') || numel(RT.rescue_mask) ~= n
                warning('mask_audit: rescue trial "%s" has no usable rescue_mask, skipped.', rt.name);
                continue
            end
            rmask = logical(RT.rescue_mask(:));
            rescue_names{end+1} = rt.name;
            col = arrayfun(@(i) 100*nnz(F(:,i) & rmask)/max(n_flagged(i),1), 1:nck)';
            pct_rescued(:, end+1) = col;
        end
        for ri = 1:numel(rescue_names)
            T_wc.(['pct_rescued_' rescue_names{ri}]) = pct_rescued(:, ri);
        end
    end

    fprintf('[mask_audit] within_channel, %d spikes total:\n', n);
    disp(T_wc);
    writetable(T_wc, fullfile(o.out_dir, 'mask_audit_wc_counts.csv'));

    % pairwise Jaccard overlap
    J = eye(nck);
    for i = 1:nck
        for j = 1:nck
            if i==j, continue; end
            a = F(:,i); b = F(:,j);
            u = nnz(a|b);
            J(i,j) = (u>0) * nnz(a&b) / max(u,1);
        end
    end

    f = figure('Visible','off','Position',[100 100 600 520],'Color','w');
    ax = axes(f);
    imagesc(ax, J, [0 1]); axis(ax,'image'); colorbar(ax); colormap(ax, parula);
    xticks(ax,1:nck); yticks(ax,1:nck); xticklabels(ax,checks); yticklabels(ax,checks);
    set(ax,'TickLabelInterpreter','none'); xtickangle(ax,40);
    for i=1:nck, for j=1:nck
        text(ax, j, i, sprintf('%.2f', J(i,j)), 'HorizontalAlignment','center', 'FontSize', 8, ...
             'Color', (J(i,j)>0.5)*[1 1 1] + (J(i,j)<=0.5)*[0 0 0]);
    end, end
    title(ax, 'within\_channel: pairwise flagged-spike overlap (Jaccard)', 'Interpreter','none');
    save_fig_local(f, fullfile(o.out_dir, 'mask_audit_wc_overlap.png'));

    has_rescue = ~isempty(rescue_names);
    f = figure('Visible','off','Position',[100 100 has_rescue*500+800 460],'Color','w');
    if has_rescue
        tl = tiledlayout(f, 1, 2, 'Padding','loose', 'TileSpacing','loose');
        ax = nexttile(tl);
    else
        ax = axes(f);
    end
    hold(ax,'on');
    bar(ax, categorical(checks), [n_flagged(:), n_unique(:)]);
    legend(ax, {'total flagged','unique to this check'}, 'Location','northoutside','Orientation','horizontal');
    ylabel(ax, '# spikes'); grid(ax,'on');
    title(ax, sprintf('within\\_channel detections (%d spikes total)', n), 'Interpreter','none');

    if has_rescue
        ax = nexttile(tl); hold(ax,'on');
        bar(ax, categorical(checks), pct_rescued);
        legend(ax, rescue_names, 'Location','northoutside','Orientation','horizontal', 'Interpreter','none');
        ylabel(ax, '% of this check''s flagged spikes rescued'); ylim(ax,[0 100]); grid(ax,'on');
        title(ax, 'rescue recovery rate per check, per sdnum', 'Interpreter','none');
    end
    save_fig_local(f, fullfile(o.out_dir, 'mask_audit_wc_counts.png'));

    % ======================= refract_viol keep_strategy =====================
    ref_ms = o.ref_ms;
    if isempty(ref_ms)
        if isfield(SPK,'par') && isfield(SPK.par,'ref') && isfield(SPK.par,'sr')
            ref_ms = SPK.par.ref / (SPK.par.sr/1000);
        else
            ref_ms = 1.5;
        end
    end
    T_ref = table(); chain_lengths = [];
    if n > 1
        gaps = diff(index_all);
        in_gap = gaps < ref_ms;
        ext = [false; in_gap; false];
        c_start = find(diff(ext)==1); c_end = find(diff(ext)==-1);
        nchains = numel(c_start);
        chain_lengths = c_end - c_start + 1;

        strategies = {'first','last','none'};
        n_removed = zeros(1,3);
        for c = 1:nchains
            n_removed(1) = n_removed(1) + (chain_lengths(c)-1);  % first: keep 1
            n_removed(2) = n_removed(2) + (chain_lengths(c)-1);  % last:  keep 1
            n_removed(3) = n_removed(3) + chain_lengths(c);      % none:  keep 0
        end
        T_ref = table(strategies(:), n_removed(:), repmat(nchains,3,1), ...
            'VariableNames', {'keep_strategy','n_spikes_removed','n_chains'});
        fprintf('\n[mask_audit] refract_viol (ref_ms=%.2f), %d chains:\n', ref_ms, nchains);
        disp(T_ref);
        writetable(T_ref, fullfile(o.out_dir, 'mask_audit_refract.csv'));

        f = figure('Visible','off','Position',[100 100 900 460],'Color','w');
        tl = tiledlayout(f,1,2,'Padding','loose','TileSpacing','loose');
        ax = nexttile(tl);
        bar(ax, categorical(strategies), n_removed); grid(ax,'on');
        ylabel(ax,'# spikes removed'); title(ax, sprintf('keep\\_strategy (%d chains)', nchains), 'Interpreter','none');
        ax = nexttile(tl);
        histogram(ax, chain_lengths, 'BinMethod','integers', 'FaceColor',[0.3 0.5 0.7]);
        xlabel(ax,'chain length (spikes)'); ylabel(ax,'# chains'); grid(ax,'on');
        title(ax,'refractory-violation chain lengths');
        save_fig_local(f, fullfile(o.out_dir, 'mask_audit_refract.png'));
    else
        fprintf('\n[mask_audit] not enough spikes for refract_viol analysis.\n');
    end

    % ======================= template cross-reference =======================
    T_tmpl = table();
    if ~isempty(o.times_file)
        TF = load(o.times_file);
        good = TF.cluster_class(:,1) ~= 0;
        class_good = TF.cluster_class(good,1)';
        spikes_tmpl = TF.spikes(good,:);
        gt_classes = unique(class_good);
        templates = zeros(numel(gt_classes), size(spikes_tmpl,2));
        template_sd = zeros(1,numel(gt_classes));
        for i = 1:numel(gt_classes)
            wf = spikes_tmpl(class_good==gt_classes(i), :);
            templates(i,:) = mean(wf,1);
            template_sd(i) = sqrt(sum(var(wf,1)));
        end
        pct_close = zeros(1,nck); n_close = zeros(1,nck);
        for i = 1:nck
            wf = spikes_all(F(:,i), :);
            if isempty(wf), continue; end
            D = sqrt(max(bsxfun(@plus, sum(wf.^2,2), sum(templates.^2,2)') - 2*(wf*templates'), 0));
            radius = o.reject_sdnum * template_sd;
            close_any = any(D <= radius, 2);
            n_close(i) = nnz(close_any);
            pct_close(i) = 100*nnz(close_any)/size(wf,1);
        end
        T_tmpl = table(checks(:), n_flagged(:), n_close(:), pct_close(:), ...
            'VariableNames', {'detection','n_flagged','n_shape_close_to_real_class','pct_shape_close'});
        fprintf('\n[mask_audit] template cross-reference (reject_sdnum=%.1f):\n', o.reject_sdnum);
        disp(T_tmpl);
        writetable(T_tmpl, fullfile(o.out_dir, 'mask_audit_template.csv'));

        f = figure('Visible','off','Position',[100 100 800 460],'Color','w');
        ax = axes(f);
        bar(ax, categorical(checks), pct_close); grid(ax,'on'); ylim(ax,[0 100]);
        ylabel(ax, sprintf('%% of flagged spikes shape-close to a real class (sd\\leq%.1f)', o.reject_sdnum));
        title(ax, 'candidate-to-loosen signal per check', 'Interpreter','none');
        save_fig_local(f, fullfile(o.out_dir, 'mask_audit_template.png'));
    end

    out = struct('wc_counts', T_wc, 'wc_overlap', J, 'wc_checks', {checks}, ...
                 'refract', T_ref, 'template_xref', T_tmpl, 'out_dir', o.out_dir, ...
                 'pct_rescued', pct_rescued, 'rescue_names', {rescue_names});
    fprintf('\n[mask_audit] wrote %s\n', o.out_dir);
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
