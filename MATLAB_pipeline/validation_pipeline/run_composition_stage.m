function run_composition_stage(cfg, arm_classes)
% RUN_COMPOSITION_STAGE  One page per arm on where its matched cluster came
% from, built from the final class labels of the Python-layout times file.
%
% Left column, top to bottom:
%   pie       II      sd1 first pass: the original SPC spikes (I, label > 0
%                     and never forced, drawn as a darker wedge inside II,
%                     area proportional to n_I/n_II) plus the forced spikes
%                     that were in the clustering pass's sd1 core for this
%                     class (sd1_pass_templates)
%             III-II  forced spikes (the sd3 pass) not in the sd1 core
%             IV      rescued spikes
%   waveforms of each pie group (I, II-I, III-II, IV), 3 per row
%   the rescued spikes (IV) by the quarantine property they failed, as % of
%   IV, % of every rescued spike (any class) and % of all quarantined spikes
%   -- each spike counted once: under its one failed mask, or 'multiple
%   masks' if it failed 2+, so the IV bars sum to 100%
%   -- the union of those properties, which is the venn's quarantined n
%
% Right column (channel-wide): waveforms of cluster 0 that was never
% quarantined, of quarantined spikes that were never rescued, and the
% quarantine properties behind the latter.
%
% Every waveform panel carries the matched (GT & MCWs) template and the sd1
% template (II), on one fixed y range shared with every other validation
% waveform figure (prepare_arm_groups / wave_ylim).
%
% II + (III-II) + IV = the whole final cluster. Spikes are counted in the
% class they ENDED in (after pass 2 and the Mahalanobis reassignment), so an
% sd1-core spike that finished in another class is not in this cluster; that
% count is noted under the chart when it is nonzero.
%
%   run_composition_stage(cfg)               % matches arm classes itself
%   run_composition_stage(cfg, arm_classes)  % reuse run_validation's matching
%
% OUTPUTS (under cfg.out_dir/cluster_composition)
%   composition_ch<ch>_<arm>.png
%   composition_counts.csv - one row per arm

if nargin < 2, arm_classes = []; end
[Gs, yl, ~, arm_classes] = prepare_arm_groups(cfg, arm_classes);

out_dir = fullfile(cfg.out_dir, 'cluster_composition');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
arms = cfg.arms;

rows = {};
for k = 1:numel(arms)
    G = Gs{k};
    cls = arm_classes(k);
    C = G.counts;
    assert(C.n_II + C.n_III_II + C.n_IV == C.n_total, 'composition slices do not add up to the cluster');

    n_quar = nnz(any(G.M(:, G.prop_active), 2));
    if n_quar ~= nnz(G.masks.q)
        warning('run_composition_stage: %s union of quarantine properties (%d) ~= quarantine.all (%d)', ...
            arms(k).name, n_quar, nnz(G.masks.q));
    end
    mb_iv = struct('labels', {G.prop_names_x}, 'active', G.prop_active_x, 'counts', sum(G.Mx(G.masks.is_IV, :), 1), ...
        'series', struct('name', {'% of rescued here (IV)', '% of all rescued', '% of all quarantined'}, ...
                         'tag',  {'IV', 'all resc.', 'quar.'}, ...
                         'n',    {C.n_IV, nnz(G.masks.resc), n_quar}, ...
                         'color',{[0.98 0.78 0.45], [0.85 0.45 0.10], [0.30 0.30 0.30]}), ...
        'title', sprintf('rescued into %s class %d by quarantine property (IV, n=%d)', arms(k).name, cls, C.n_IV));
    mb_qnr = struct('labels', {G.prop_names_x}, 'active', G.prop_active_x, 'counts', sum(G.Mx(G.masks.is_qnr, :), 1), ...
        'series', struct('name', {'% of quarantined, not rescued', '% of all quarantined'}, ...
                         'tag',  {'QnR', 'quar.'}, ...
                         'n',    {nnz(G.masks.is_qnr), n_quar}, ...
                         'color',{[0.95 0.6 0.1], [0.30 0.30 0.30]}), ...
        'title', sprintf('quarantined, never rescued, by quarantine property (n=%d)', nnz(G.masks.is_qnr)));

    f = figure('Visible','off','Position',[100 100 1700 1750],'Color','w');

    % ---- left column: pie ----
    ax = axes(f, 'Position', [0.10 0.665 0.28 0.27]);
    draw_composition_pie(ax, C);

    annotation(f, 'textbox', [0 0.955 1 0.04], 'EdgeColor', 'none', 'FontSize', 14, 'FontWeight', 'bold', ...
        'Interpreter', 'none', 'HorizontalAlignment', 'center', ...
        'String', sprintf('ch%d  %s class %d  (n=%d)', cfg.ch, arms(k).name, cls, C.n_total));
    % only worth a note when sd1-core spikes actually left the class
    if C.n_core_moved > 0
        annotation(f, 'textbox', [0.01 0.615 0.48 0.03], 'EdgeColor', 'none', 'FontSize', 9, ...
            'Interpreter', 'none', 'HorizontalAlignment', 'center', 'String', ...
            sprintf(['%d of the %d sd1-core spikes for class %d ended in another class ' ...
                     '(pass 2 / Mahalanobis reassignment) and are not in this pie'], C.n_core_moved, C.n_sd1_core, cls));
    end

    % ---- left column: waveforms of the pie groups, then the IV bar ----
    draw_wave_grid(f, [0.05 0.345 0.43 0.25], G.comp, G.templates, yl, 3);
    plot_mask_bar(axes(f, 'Position', [0.05 0.035 0.43 0.215]), mb_iv);

    % ---- right column: channel-wide groups ----
    draw_wave_group(axes(f, 'Position', [0.56 0.69 0.41 0.23]), G.c0, G.templates, yl, true);
    draw_wave_group(axes(f, 'Position', [0.56 0.38 0.41 0.23]), G.qnr, G.templates, yl, true);
    plot_mask_bar(axes(f, 'Position', [0.56 0.035 0.41 0.215]), mb_qnr);

    print(f, fullfile(out_dir, sprintf('composition_ch%d_%s.png', cfg.ch, arms(k).name)), '-dpng', '-r130');
    close(f);

    fprintf('[composition] %s class %d: II=%d (I=%d + core %d)  III-II=%d  IV=%d  total=%d  (core moved out %d)\n', ...
        arms(k).name, cls, C.n_II, C.n_I, C.n_core, C.n_III_II, C.n_IV, C.n_total, C.n_core_moved);
    rows(end+1,:) = {arms(k).name, cls, C.n_I, C.n_II, C.n_III_II, C.n_IV, C.n_total, C.n_sd1_core, C.n_core_moved}; %#ok<AGROW>
end

writetable(cell2table(rows, 'VariableNames', {'arm','class','n_I_original_spc','n_II_sd1_first_pass', ...
    'n_III_minus_II_forced','n_IV_rescued','n_total','n_sd1_core','n_sd1_core_moved_out'}), ...
    fullfile(out_dir, 'composition_counts.csv'));
end
