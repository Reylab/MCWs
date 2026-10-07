function T = rescue_mask_breakdown(input, varargin)
% RESCUE_MASK_BREAKDOWN  Attribute rescued spikes to the quarantine mask(s) they
%                        originally failed, and visualise the overlap.
%
% After rescue_spikes has run, every rescued spike is (by construction) a spike
% that failed at least one quarantine mask before being re-admitted by template
% matching. This tool loads the per-spike masks from the spikes file and, for the
% spikes flagged in rescue_mask, works out which mask(s) each one failed:
%
%   mask_nonart          FALSE -> spike is a cross-channel / bundle artifact
%   mask_non_quarantine  FALSE -> spike failed within-channel waveform QC
%   mask_non_refract     FALSE -> spike is in a refractory-violation chain
%
% For each times file it produces one figure:
%   (1) Venn diagram of the rescued spikes over the 2-3 masks in play.
%   (2) Per-region bar chart: quarantined-but-not-rescued vs rescued, with the
%       rescue rate for each disjoint region.
%   (3) Per-cluster stacked bar: for each rescued cluster, how many of its
%       rescued spikes involved each mask (a spike failing two masks is counted
%       under both -> "mask involvement", not disjoint).
%
% A pooled figure over all processed files is drawn at the end when >1 file.
%
% Usage:
%   rescue_mask_breakdown(channels)
%   rescue_mask_breakdown('all')
%   rescue_mask_breakdown(channels, 'folder_name', '.')
%   rescue_mask_breakdown(channels, 'save_fig', true, 'fig_dir', 'rescue_breakdown')
%
% Input (same semantics as rescue_spikes):
%   numeric vector of channel IDs | 'all' | cell array of times_*.mat filenames
%
% Optional name-value:
%   'folder_name'   ''      times folder override, relative to pwd ('.' = pwd)
%   'spikes_folder' ''      spikes folder override (default: root_dir/spikes*)
%   'save_fig'      true    write PNGs
%   'fig_dir'       'rescue_breakdown'   output subfolder (under the times folder)
%   'show_figures'  true    leave figures visible
%
% Output:
%   T  - table, one row per (file, disjoint mask-region) with n_quarantined,
%        n_rescued and rescue_rate. Also written as rescue_mask_breakdown.csv.

% ---- Parse args ----------------------------------------------------------
p = inputParser;
addParameter(p, 'folder_name',   '',                 @ischar);
addParameter(p, 'spikes_folder', '',                 @ischar);
addParameter(p, 'save_fig',      true,               @islogical);
addParameter(p, 'fig_dir',       'rescue_breakdown', @ischar);
addParameter(p, 'show_figures',  true,               @islogical);
parse(p, varargin{:});
folder_name      = p.Results.folder_name;
spikes_folder_in = p.Results.spikes_folder;
save_fig         = p.Results.save_fig;
fig_dir_name     = p.Results.fig_dir;
visstr           = 'on'; if ~p.Results.show_figures, visstr = 'off'; end

valid_masks  = {'mask_nonart', 'mask_non_quarantine', 'mask_non_refract'};
mask_labels  = {'artifact', 'waveform QC', 'refractory'};
mask_colors  = [0.20 0.55 0.85; 0.90 0.42 0.30; 0.34 0.70 0.38];

% ---- Resolve folders (mirrors rescue_spikes) ----------------------------
root_dir = resolve_session_root();
[~, current_dir_name] = fileparts(pwd);

if ~isempty(folder_name)
    active_times_dir = fullfile(pwd, folder_name);
    if ~exist(active_times_dir, 'dir')
        error('rescue_mask_breakdown: folder_name "%s" does not exist.', folder_name);
    end
elseif contains(current_dir_name, 'merge') || contains(current_dir_name, 'times_')
    active_times_dir = pwd;
else
    d = dir(fullfile(pwd, 'times*')); d = d([d.isdir]);
    if isempty(d)
        error('rescue_mask_breakdown: no times* folder in %s. Use ''folder_name''.', pwd);
    end
    [~, ix] = max([d.datenum]);
    active_times_dir = fullfile(pwd, d(ix).name);
    fprintf('Auto-detected times folder: %s\n', d(ix).name);
end

if ~isempty(spikes_folder_in)
    active_spikes_dir = spikes_folder_in;
else
    ds = dir(fullfile(root_dir, 'spikes*')); ds = ds([ds.isdir]);
    if isempty(ds)
        error('rescue_mask_breakdown: no spikes* folder in %s. Use ''spikes_folder''.', root_dir);
    end
    [~, ixs] = max([ds.datenum]);
    active_spikes_dir = fullfile(root_dir, ds(ixs).name);
end

% ---- File list --------------------------------------------------------------
file_list = resolve_times_list(input, active_times_dir);
if isempty(file_list)
    error('rescue_mask_breakdown: no matching times_*.mat files.');
end

if save_fig
    fig_out_dir = fullfile(active_times_dir, fig_dir_name);
    if ~exist(fig_out_dir, 'dir'), mkdir(fig_out_dir); end
else
    fig_out_dir = '';
end

% ---- Process --------------------------------------------------------------
rows        = {};   % accumulates table rows
pooled_keys = {};
pooled_e    = struct('masks_used', {}, 'code', {}, 'n_quar', {}, 'n_resc', {});

for k = 1:numel(file_list)
    r = process_one(file_list{k}, active_spikes_dir, valid_masks, mask_labels, ...
                    mask_colors, visstr, fig_out_dir);
    if isempty(r), continue; end

    for j = 1:numel(r.region_code)
        rows(end+1, :) = { r.ch_lbl, r.region_label{j}, ...
                           r.n_quar(j), r.n_resc(j), r.rescue_rate(j) };
    end

    % pool by identical mask-name-set
    key  = strjoin(r.masks_used, '+');
    slot = find(strcmp(pooled_keys, key), 1);
    if isempty(slot)
        pooled_keys{end+1} = key;
        ne = struct();
        ne.masks_used = r.masks_used;
        ne.code       = r.region_code;
        ne.n_quar     = r.n_quar;
        ne.n_resc     = r.n_resc;
        pooled_e(end+1) = ne;
    else
        e = pooled_e(slot);
        [tf, loc] = ismember(r.region_code, e.code);
        e.n_quar(loc(tf)) = e.n_quar(loc(tf)) + r.n_quar(tf);
        e.n_resc(loc(tf)) = e.n_resc(loc(tf)) + r.n_resc(tf);
        new_codes = r.region_code(~tf);
        if ~isempty(new_codes)
            e.code   = [e.code;   new_codes];
            e.n_quar = [e.n_quar; r.n_quar(~tf)];
            e.n_resc = [e.n_resc; r.n_resc(~tf)];
        end
        pooled_e(slot) = e;
    end
end

var_names = {'channel', 'mask_region', 'n_quarantined', 'n_rescued', 'rescue_rate'};
if isempty(rows)
    fprintf('rescue_mask_breakdown: no rescued spikes in any processed file.\n');
    T = cell2table(cell(0, numel(var_names)), 'VariableNames', var_names);
    return;
end
T = cell2table(rows, 'VariableNames', var_names);

if ~isempty(fig_out_dir)
    csv_path = fullfile(fig_out_dir, 'rescue_mask_breakdown.csv');
    writetable(T, csv_path);
    fprintf('Wrote %s\n', csv_path);
end

% ---- Pooled figure -------------------------------------------------------
if numel(file_list) > 1
    for slot = 1:numel(pooled_keys)
        e = pooled_e(slot);
        n = numel(e.masks_used);
        labs = mask_labels(ismember(valid_masks, e.masks_used));
        cols = mask_colors(ismember(valid_masks, e.masks_used), :);

        f = figure('Visible', visstr, 'Name', ['pooled: ' pooled_keys{slot}], ...
                   'Units', 'normalized', 'OuterPosition', [0.05 0.1 0.6 0.7]);
        tiledlayout(f, 1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

        nexttile;
        draw_venn(gca, labs, cols, e.code, e.n_resc, n);
        title(sprintf('Pooled rescued spikes  (n = %d)', sum(e.n_resc)));

        nexttile;
        region_bar(gca, e.code, e.n_quar, e.n_resc, labs);
        title('Pooled: quarantined vs rescued per region');

        sgtitle(sprintf('Rescue mask breakdown — pooled over %d files [%s]', ...
                numel(file_list), strjoin(labs, ', ')), 'Interpreter', 'none');

        if ~isempty(fig_out_dir)
            saveas(f, fullfile(fig_out_dir, sprintf('pooled_%s.png', ...
                   matlab.lang.makeValidName(pooled_keys{slot}))));
        end
    end
end

end % ======================================================================


function r = process_one(times_file, spikes_dir, valid_masks, mask_labels, ...
                         mask_colors, visstr, fig_out_dir)
r = [];
[~, fname_noext] = fileparts(times_file);
ch_lbl = regexprep(fname_noext, '^times_', '');
spike_file = fullfile(spikes_dir, [ch_lbl '_spikes.mat']);

if ~exist(times_file, 'file')
    fprintf('  [%s] times file missing, skipping.\n', ch_lbl); return;
end
T = load(times_file);
if ~isfield(T, 'rescue_mask') || isempty(T.rescue_mask) || ~any(T.rescue_mask)
    fprintf('  [%s] no rescue_mask (or nothing rescued), skipping.\n', ch_lbl); return;
end
if ~exist(spike_file, 'file')
    fprintf('  [%s] spikes file missing (%s), skipping.\n', ch_lbl, spike_file); return;
end
SPK = load(spike_file);
if ~isfield(SPK, 'index_all')
    fprintf('  [%s] spikes file has no index_all, skipping.\n', ch_lbl); return;
end

index_all   = reshape(double(SPK.index_all), 1, []);
n_all       = numel(index_all);
rescue_mask = logical(reshape(T.rescue_mask, 1, []));
if numel(rescue_mask) ~= n_all
    fprintf('  [%s] rescue_mask length (%d) != index_all (%d), skipping.\n', ...
            ch_lbl, numel(rescue_mask), n_all); return;
end

% Which masks defined the quarantine for this file
if isfield(T, 'quarantine_masks_used') && ~isempty(T.quarantine_masks_used)
    masks_used = cellstr(T.quarantine_masks_used);
    masks_used = masks_used(ismember(masks_used, valid_masks));
else
    masks_used = valid_masks(isfield(SPK, valid_masks));
end
masks_used = valid_masks(ismember(valid_masks, masks_used));   % canonical order
n_masks = numel(masks_used);
if n_masks == 0
    fprintf('  [%s] no usable masks, skipping.\n', ch_lbl); return;
end

labs = mask_labels(ismember(valid_masks, masks_used));
cols = mask_colors(ismember(valid_masks, masks_used), :);

% Per-spike FAIL matrix (TRUE = failed that mask = reason for quarantine)
F = false(n_all, n_masks);
for m = 1:n_masks
    mv = logical(reshape(SPK.(masks_used{m}), 1, []));
    if numel(mv) ~= n_all
        fprintf('  [%s] mask %s length mismatch, skipping.\n', ch_lbl, masks_used{m}); return;
    end
    F(:, m) = ~mv(:);
end

quar = any(F, 2);                 % quarantined spikes
resc = rescue_mask(:);            % rescued spikes (subset of quar)

% Disjoint region code per spike: bit m set => failed mask m
w    = 2.^(0:n_masks-1);
code = F * w(:);                  % 0 = passed everything

codes_present = (1:(2^n_masks - 1))';
n_quar = arrayfun(@(c) sum(code(quar) == c), codes_present);
n_resc = arrayfun(@(c) sum(code(resc) == c), codes_present);
keep   = (n_quar + n_resc) > 0;
codes_present = codes_present(keep);
n_quar = n_quar(keep);
n_resc = n_resc(keep);
rate   = n_resc ./ max(n_quar, 1);

region_label = arrayfun(@(c) code_to_label(c, labs), codes_present, 'UniformOutput', false);

% ---- Per-cluster mask involvement (non-disjoint) ----
% Map rescued spikes -> post-rescue cluster via spike time.
resc_pos   = find(resc);
resc_time  = index_all(resc_pos);
cl_of_resc = nan(size(resc_pos));
if isfield(T, 'cluster_class') && ~isempty(T.cluster_class)
    cc_t = T.cluster_class(:, 2);
    cc_c = T.cluster_class(:, 1);
    [tf, loc] = ismember(resc_time(:), cc_t);
    cl_of_resc(tf) = cc_c(loc(tf));
end
resc_clusters = unique(cl_of_resc(~isnan(cl_of_resc) & cl_of_resc ~= 0));
involve = zeros(numel(resc_clusters), n_masks);   % cluster x mask counts
for ci = 1:numel(resc_clusters)
    sel = cl_of_resc == resc_clusters(ci);
    involve(ci, :) = sum(F(resc_pos(sel), :), 1);
end

% ---- Figure ----
f = figure('Visible', visstr, 'Name', ch_lbl, ...
           'Units', 'normalized', 'OuterPosition', [0.03 0.08 0.72 0.8]);
tl = tiledlayout(f, 1, 3, 'Padding', 'compact', 'TileSpacing', 'compact');

nexttile(tl);
draw_venn(gca, labs, cols, codes_present, n_resc, n_masks);
title(sprintf('Rescued spikes by mask  (n = %d)', sum(n_resc)));

nexttile(tl);
region_bar(gca, codes_present, n_quar, n_resc, labs);
title('Quarantined vs rescued per region');

ax3 = nexttile(tl);
if ~isempty(resc_clusters)
    nC = numel(resc_clusters);
    plot_mat = involve;
    if nC == 1, plot_mat = [involve; zeros(1, n_masks)]; end   % 1-row 'stacked' quirk
    b = barh(ax3, 1:size(plot_mat, 1), plot_mat, 'stacked');
    for m = 1:n_masks, b(m).FaceColor = cols(m, :); end
    ylim(ax3, [0.5, nC + 0.5]);
    set(ax3, 'YTick', 1:nC, ...
        'YTickLabel', arrayfun(@(c) sprintf('C%d', c), resc_clusters, 'UniformOutput', false));
    xlabel(ax3, 'rescued spikes involving mask');
    legend(ax3, labs, 'Location', 'southoutside', 'Orientation', 'horizontal', 'Interpreter', 'none');
    title(ax3, 'Per-cluster mask involvement');
    grid(ax3, 'on'); box(ax3, 'off');
else
    axis(ax3, 'off');
    text(ax3, 0.5, 0.5, 'no cluster mapping', 'HorizontalAlignment', 'center');
end

n_resc_total = sum(resc);
n_quar_total = sum(quar);
sgtitle(sprintf('%s  —  rescued %d / %d quarantined (%.0f%%)  [masks: %s]', ...
        ch_lbl, n_resc_total, n_quar_total, 100 * n_resc_total / max(n_quar_total, 1), ...
        strjoin(labs, ', ')), 'Interpreter', 'none');

if ~isempty(fig_out_dir)
    saveas(f, fullfile(fig_out_dir, sprintf('%s_rescue_mask_breakdown.png', ch_lbl)));
end

% ---- Return struct ----
r = struct();
r.ch_lbl       = ch_lbl;
r.masks_used   = masks_used;
r.region_code  = codes_present;
r.region_label = region_label;
r.n_quar       = n_quar;
r.n_resc       = n_resc;
r.rescue_rate  = rate;
end % process_one


% ========================================================================
%  Helpers
% ========================================================================
function lbl = code_to_label(c, labs)
bits = find(bitget(c, 1:numel(labs)));
if numel(bits) == 1
    lbl = [labs{bits} ' only'];
else
    lbl = strjoin(labs(bits), ' & ');
end
end


function region_bar(ax, codes, n_quar, n_resc, labs)
axes(ax);
labels = arrayfun(@(c) code_to_label(c, labs), codes, 'UniformOutput', false);
data   = [n_resc(:), max(n_quar(:) - n_resc(:), 0)];   % rescued | still quarantined
b = bar(1:numel(codes), data, 'stacked');
b(1).FaceColor = [0.34 0.70 0.38];
b(2).FaceColor = [0.75 0.75 0.75];
set(ax, 'XTick', 1:numel(codes), 'XTickLabel', labels, 'XTickLabelRotation', 30);
ylabel('spike count'); grid on; box off;
legend({'rescued', 'not rescued'}, 'Location', 'best');
ymax = max(n_quar(:)); if isempty(ymax) || ymax == 0, ymax = 1; end
ylim(ax, [0, ymax * 1.18]);
for i = 1:numel(codes)
    if n_quar(i) > 0
        text(i, n_quar(i) + ymax * 0.03, sprintf('%.0f%%', 100 * n_resc(i) / n_quar(i)), ...
            'HorizontalAlignment', 'center', 'FontSize', 8, 'FontWeight', 'bold');
    end
end
end


function draw_venn(ax, labs, cols, codes, counts, n_masks)
% Fixed-geometry (non area-proportional) 2- or 3-set Venn with counts placed in
% each region. `codes` are the bitmask region ids present, `counts` the value for
% each. n_masks in {1,2,3}.
axes(ax);
cla(ax); hold(ax, 'on'); axis(ax, 'equal'); axis(ax, 'off');
th = linspace(0, 2*pi, 200);

cmap = containers.Map('KeyType', 'double', 'ValueType', 'double');
for i = 1:numel(codes), cmap(codes(i)) = counts(i); end

switch n_masks
    case 1
        r = 1.0; cx = 0; cy = 0;
        patch(ax, cx + r*cos(th), cy + r*sin(th), cols(1, :), 'FaceAlpha', 0.35, 'EdgeColor', cols(1, :), 'LineWidth', 1.5);
        text(ax, 0, 0, num2str(safeget(cmap, 1)), 'HorizontalAlignment', 'center', 'FontSize', 13, 'FontWeight', 'bold');
        text(ax, 0, r + 0.15, labs{1}, 'HorizontalAlignment', 'center', 'FontSize', 11, 'Interpreter', 'none');
        xlim(ax, [-1.6 1.6]); ylim(ax, [-1.6 1.6]);

    case 2
        r = 0.95; C = [-0.5 0; 0.5 0];
        for m = 1:2
            patch(ax, C(m,1) + r*cos(th), C(m,2) + r*sin(th), cols(m, :), ...
                  'FaceAlpha', 0.35, 'EdgeColor', cols(m, :), 'LineWidth', 1.5);
        end
        tpos = [-0.95 0; 0.95 0; 0 0];                 % A, B, AB
        tcode = [1 2 3];
        for i = 1:3
            text(ax, tpos(i,1), tpos(i,2), num2str(safeget(cmap, tcode(i))), ...
                'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
        end
        text(ax, C(1,1), r + 0.18, labs{1}, 'HorizontalAlignment', 'center', 'FontSize', 11, 'Interpreter', 'none');
        text(ax, C(2,1), r + 0.18, labs{2}, 'HorizontalAlignment', 'center', 'FontSize', 11, 'Interpreter', 'none');
        xlim(ax, [-2 2]); ylim(ax, [-1.6 1.8]);

    otherwise % 3
        r = 1.0; C = [-0.5 0.35; 0.5 0.35; 0 -0.5];
        for m = 1:3
            patch(ax, C(m,1) + r*cos(th), C(m,2) + r*sin(th), cols(m, :), ...
                  'FaceAlpha', 0.30, 'EdgeColor', cols(m, :), 'LineWidth', 1.5);
        end
        tpos  = [-1.05 0.80;  1.05 0.80;  0 0.95; ...
                  0.00 -1.25; -0.78 -0.35; 0.78 -0.35; ...
                  0.00 -0.05];
        tcode = [1 2 3 4 5 6 7];
        for i = 1:7
            text(ax, tpos(i,1), tpos(i,2), num2str(safeget(cmap, tcode(i))), ...
                'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold');
        end
        text(ax, -1.25, 1.30, labs{1}, 'HorizontalAlignment', 'center', 'FontSize', 11, 'Interpreter', 'none');
        text(ax,  1.25, 1.30, labs{2}, 'HorizontalAlignment', 'center', 'FontSize', 11, 'Interpreter', 'none');
        text(ax,  0.00, -1.65, labs{3}, 'HorizontalAlignment', 'center', 'FontSize', 11, 'Interpreter', 'none');
        xlim(ax, [-2 2]); ylim(ax, [-2 1.8]);
end
hold(ax, 'off');
end


function v = safeget(m, k)
if isKey(m, k), v = m(k); else, v = 0; end
end


% ========================================================================
%  File list resolver (subset of rescue_spikes' resolver)
% ========================================================================
function file_list = resolve_times_list(input, times_dir)
file_list = {};
if ischar(input) || isstring(input)
    input = char(input);
    if strcmp(input, 'all')
        d = dir(fullfile(times_dir, 'times_*.mat'));
        file_list = cellfun(@(n) fullfile(times_dir, n), {d.name}, 'UniformOutput', false);
        return;
    end
    if exist(input, 'file'), file_list = {input};
    elseif exist(fullfile(times_dir, input), 'file'), file_list = {fullfile(times_dir, input)};
    end
    return;
end
if iscell(input)
    for k = 1:numel(input)
        f = input{k};
        if exist(f, 'file'), file_list{end+1} = f;
        elseif exist(fullfile(times_dir, f), 'file'), file_list{end+1} = fullfile(times_dir, f);
        else, warning('rescue_mask_breakdown: not found: %s', f);
        end
    end
    return;
end
if isnumeric(input)
    d = dir(fullfile(times_dir, 'times_*.mat'));
    names = {d.name};
    for ci = input(:)'
        hit = false;
        for k = 1:numel(names)
            tok = regexp(names{k}, '(\d+)\.mat$', 'tokens', 'once');
            if ~isempty(tok) && str2double(tok{1}) == ci
                file_list{end+1} = fullfile(times_dir, names{k});
                hit = true; break;
            end
        end
        if ~hit, warning('rescue_mask_breakdown: no times_*.mat for channel %d', ci); end
    end
    return;
end
error('rescue_mask_breakdown: unrecognized input type.');
end
