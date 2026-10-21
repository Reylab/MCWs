function fig_cells = loop_plot_responses_BCM_response_window(data_table, grapes, ss_num, max_nwins, ...
                                                    rank_config, ejex, save_fig, extra_label, ...
                                                    close_fig, order_offset, ...
                                                    priority_chs_ranking, is_ch_grapes, ...
                                                    b_yellow_patch, b_short_win, ...
                                                    b_img_lbl_legend, varargin)
    % MODIFIED COPY of loop_plot_responses_BCM_online.m: everything is
    % identical EXCEPT plot_rasters/plot_ifr also draw two lines at
    % data_table.win_lo(stim_idx)/data_table.win_hi(stim_idx) -- the caller's
    % own "in vs out" classification boundary (fixed +/-100ms or dynamic
    % onset+dura+buffer), which is NOT the same as the existing onset/dura or
    % win_cent markers already drawn here. data_table MUST have win_lo/win_hi
    % columns (one row per stimulus, same order/length as everything else).

    ipr = inputParser;
    addParameter(ipr, 'image_cache', [],  @(x) isempty(x) || isa(x,'containers.Map'));
    addParameter(ipr, 'output_dir',  pwd, @ischar);
    addParameter(ipr, 'clip_to_window', false, @islogical);
    % x_jitter: half-width of the uniform x-axis jitter applied to the per-
    % stimulus dots/lines of the GT-vs-ARM paired panels (0 = off, the
    % original behavior). Points with equal values otherwise sit exactly on
    % top of each other at x=1 (GT) / x=2 (ARM).
    addParameter(ipr, 'x_jitter', 0, @(x) isnumeric(x) && isscalar(x) && x >= 0);
    parse(ipr, varargin{:});
    image_cache = ipr.Results.image_cache;
    output_dir  = ipr.Results.output_dir;
    clip_to_window = ipr.Results.clip_to_window;
    x_jitter = ipr.Results.x_jitter;

    if ~exist('b_img_lbl_legend', 'var')
        b_img_lbl_legend = false;
    end
    if ~exist('b_short_win', 'var')
        b_short_win = false;
    end
    if ~exist('b_yellow_patch', 'var')
        b_yellow_patch = true;
    end
    if ~exist('order_offset','var') || isempty(order_offset)
        order_offset=0;
    end
    if ~exist('is_ch_grapes', 'var')         is_ch_grapes         = false; end
    if ~exist('priority_chs_ranking', 'var') priority_chs_ranking = []; end

    % 4 sub-rows per stimulus (image, raster, PSTH, + a GT-vs-ARM actual-value
    % bar -- the last one only actually drawn when data_table carries the
    % gt_* reference columns it needs, see plot_arm_gt_value_bar), was 3
    % originally.
    % (A "cluster total vs response count" 5th/4th bar row used to live here
    % too, but that was a whole subplot spent on ONE number that never
    % changed across a page -- it's now folded into the image panel's own
    % mu/std text line instead, see plot_img_zscore_class.)
    cols = 5;
    % 4th sub-row = GT-vs-ARM value bar; only when there is a GT to compare
    % against (gt_ref joined) -- GT's own pages and subset pages have none
    has_gt = ismember('gt_median_post', data_table.Properties.VariableNames);
    subrows_per_stim = 3 + has_gt;
    stim_cap = 20;   % max stimuli placed on one page (4 groups of `cols`), same pagination as before

    nwins= min(max_nwins,ceil(height(data_table)/stim_cap));

    warning off

    if ~exist('save_fig','var'), save_fig = true; end
    if ~exist('extra_label','var'), extra_label = ''; end
    if ~exist('close_fig','var'), close_fig = false; end

    set(groot,'defaultaxesfontsmoothing','off')
    set(groot,'defaultfiguregraphicssmoothing','off')
    set(groot,'defaultaxestitlefontsizemultiplier',1.1)
    set(groot,'defaultaxestitlefontweight','normal')
    set(groot,{'DefaultAxesXColor','DefaultAxesYColor','DefaultAxesZColor'},{'k','k','k'})

    grid_w = 798;   % unchanged physical width of the per-stimulus subplot grid
    row_px = 80;   % physical pixels per sub-row, held constant regardless of
                   % how many stimulus row-groups a given PAGE actually needs
    % Right-side margin, reserved once per page for a shared legend (the
    % med/mean/zscore/resp colors used in plot_arm_gt_value_bar) and the
    % median_post GT-vs-ARM summary panel (see draw_page_margin) -- added
    % ON TOP of grid_w so the existing per-stimulus grid keeps its exact
    % pixel width instead of shrinking to make room.
    margin_frac = 0.16;
    fig_w = grid_w / (1 - margin_frac);

    screen_size = get(0, 'ScreenSize');
    screen_width = screen_size(3);
    screen_height = screen_size(4);

    fontsize_mult = 1.4;

    fig_cells = cell(nwins,1);
    for nwin = 1:nwins
        % Compute THIS page's real content first (before creating the figure
        % or the subplot grid) so the figure is only as tall as the stimuli
        % it actually holds -- a page with, say, 8 stimuli (1 full row-group
        % of 5 + a partial group of 3) used to always get a fixed 20-stimuli-
        % capacity (4 group) figure height regardless of real content,
        % leaving most of the page blank.
        stims2plot = (1:stim_cap)+(nwin-1)*stim_cap+order_offset;
        real_max_stims = min(stim_cap, size(data_table,1) - order_offset - (nwin-1)*stim_cap);
        n_groups = max(1, ceil(real_max_stims/cols));
        rows = n_groups * subrows_per_stim;
        [pic_num, rast_num, hist_num, pct_num] = build_row_indices(n_groups, cols, subrows_per_stim);
        fig_h = row_px * rows;

        if nwin <=3
            pos_x = 10 + (nwin-1) * fig_w;
            pos_y = screen_height / 2;
        else
            pos_x = 10 + (nwin-4) * fig_w;
            pos_y = 0;
        end
        pos = [pos_x, pos_y, fig_w, fig_h];

        fig = figure('Visible','off','Position', pos,...
            'Units','pixels','GraphicsSmoothing','off','Color',[1,1,1],...
            'Name',sprintf('Response window: %d',nwin));

        labels = {'same_feats', 'diff_feats', 'all_but_x'};
        colors = {'magenta', 'blue', 'cyan'};
        if ~contains(grapes.ImageNames.folder{1},'categ_localiz')
            labels = {'same_unit', 'same_cat'};
            colors = {'magenta', 'blue'};
        end
        if b_img_lbl_legend
            x0 = 0.02; y0 = 0.98; w = 0.01; h = 0.01; dx = 0.12;
            for k = 1:numel(labels)
                annotation(fig, 'rectangle', [x0+(k-1)*dx, y0, w, h], 'FaceColor', colors{k}, 'EdgeColor', colors{k});
                annotation(fig, 'textbox', [x0+(k-1)*dx+w+0.005, y0, 0.07, h], 'String', labels{k}, ...
                            'Interpreter','none', 'LineStyle', 'none', 'FontSize', 8, ...
                            'VerticalAlignment', 'middle', 'Color', colors{k});
            end
        end

        if is_ch_grapes
            ifr_y_max = max(cellfun(@(x) max(x(:),[],"omitnan"), data_table(stims2plot(1:real_max_stims),:).IFR),[],'omitnan');
        else
            ifr_y_max = -1;
        end
        for i = 1:real_max_stims
            stim_idx = stims2plot(i);

            subplot_ax = subplot(rows,cols,pic_num(i),'align');
            scale_factor = plot_img_zscore_class(stim_idx, data_table, grapes, ...
                                                    subplot_ax, fontsize_mult, image_cache);

            if ~is_ch_grapes
                add_bundle_label(stim_idx, data_table, grapes, ...
                                 priority_chs_ranking, fontsize_mult, ...
                                 scale_factor);
            end

            set(subplot_ax,'xtick',[],'ytick',[]);box off;
            shrink_axes_gap(subplot_ax, 0.72);
            compress_to_grid_area(subplot_ax, margin_frac);

            subplot_ax = subplot(rows,cols,rast_num(i),'align');
            plot_rasters(stim_idx, data_table, grapes, rank_config, ...
                            subplot_ax, b_yellow_patch, b_short_win, clip_to_window);
            if b_short_win
                time_pre_ms = grapes.time_pre / 2;
                time_pos_ms = grapes.time_pos / 2;
                scale_ax(subplot_ax, 0.5);
            else
                time_pre_ms = grapes.time_pre;
                time_pos_ms = grapes.time_pos;
            end
            shrink_axes_gap(subplot_ax, 0.72);
            compress_to_grid_area(subplot_ax, margin_frac);
            % nudge the raster down into the free space above its IFR plot,
            % clear of the image panel's text lines (mu/sd, counts, sign p)
            rpos = get(subplot_ax, 'Position');
            rpos(2) = rpos(2) - 0.25*rpos(4);
            set(subplot_ax, 'Position', rpos);

            subplot_ax = subplot(rows,cols,hist_num(i),'align');
            plot_ifr(stim_idx, data_table, grapes, ejex, ifr_y_max, ...
                                subplot_ax, fontsize_mult, b_short_win)
            % Only the stimuli in the page's actual LAST row-group of `cols`
            % (not just "the last `cols` stimuli overall") get tick labels --
            % with multiple row-groups stacked per page (e.g. 8 stimuli = a
            % full group of 5 + a partial group of 3), "i > real_max_stims-cols"
            % wrongly also caught the tail columns of an EARLIER, non-bottom
            % group, whose rotated tick labels then bled into that group's own
            % bar row underneath -- exactly the overlap this run surfaced.
            this_group = ceil(i/cols);
            last_group = ceil(real_max_stims/cols);
            if this_group == last_group
                set(subplot_ax,'xtick',[-time_pre_ms 0 time_pos_ms/2 time_pos_ms]);
            else
                axis off
            end
            if b_short_win
                scale_ax(subplot_ax, 0.5);
            end
            % Only the last group's PSTH row actually draws tick labels (see
            % above), and those labels sit BELOW the axes box, in the same
            % gap the row-vs-row-below shrink is supposed to protect --
            % give that specific row extra bottom margin so its own labels
            % don't bleed into the GT-vs-ARM panel directly underneath it,
            % while every other (label-less) PSTH row keeps the normal gap.
            if this_group == last_group
                shrink_axes_gap(subplot_ax, 0.60);
            else
                shrink_axes_gap(subplot_ax, 0.72);
            end
            compress_to_grid_area(subplot_ax, margin_frac);

            if has_gt
                subplot_ax = subplot(rows,cols,pct_num(i),'align');
                plot_arm_gt_value_bar(stim_idx, data_table, subplot_ax, fontsize_mult);
                shrink_axes_gap(subplot_ax, 0.72);
                compress_to_grid_area(subplot_ax, margin_frac);
            end
        end

        draw_page_margin(fig, data_table, margin_frac, fontsize_mult, x_jitter);

        add_figure_title(grapes, rank_config, ss_num, nwin, extra_label)

        if save_fig
            extra_label_name = extra_label;
            aaa=strfind(extra_label,' (');
            if ~isempty(aaa)
                extra_label_name = extra_label(1:aaa-1);
            end
            fig_filename = sprintf('%s_resp_window_subscr %d_fig %d.png',extra_label_name,ss_num,nwin);
            fig_filename = fullfile(output_dir, fig_filename);
            try
                print(fig, fig_filename, '-dpng', '-r150');
            catch
                print(fig,'-dpng','-r150', fig_filename);
            end
        end

        if close_fig
            close(fig)
        else
            fig_cells{nwin} = fig;
        end
    end

    if save_fig
        extra_label_name = extra_label;
        aaa = strfind(extra_label,' (');
        if ~isempty(aaa)
            extra_label_name = extra_label(1:aaa-1);
        end
        plot_gt_arm_summary_figure(data_table, output_dir, extra_label_name, fontsize_mult, x_jitter);
        plot_sign_test_figure(data_table, output_dir, extra_label_name, fontsize_mult);
    end

    warning on
    set(groot,'defaultfiguregraphicssmoothing','remove')
    set(groot,'defaultaxesfontsmoothing','remove')
    set(groot,'defaultaxestitlefontsizemultiplier','remove')
    set(groot,'defaultaxestitlefontweight','remove')
    set(groot,{'DefaultAxesXColor','DefaultAxesYColor','DefaultAxesZColor'},{'remove','remove','remove'})

end
function [pic_num, rast_num, hist_num, pct_num] = build_row_indices(n_groups, cols, subrows_per_stim)
% BUILD_ROW_INDICES  subplot() linear indices for N_GROUPS stimulus
% row-groups (each COLS stimuli wide, SUBROWS_PER_STIM grid-rows tall:
% image/raster/PSTH/GT-vs-ARM-value-bar, in that top-to-bottom order), so the caller can
% size its subplot(rows,cols,...) grid to exactly how many groups THIS page
% actually needs instead of a fixed worst-case capacity.
    pic_num = zeros(1, n_groups*cols); rast_num = pic_num; hist_num = pic_num; pct_num = pic_num;
    for g = 0:n_groups-1
        base = g*subrows_per_stim*cols;
        sel = g*cols+(1:cols);
        pic_num(sel)  = base + (1:cols);
        rast_num(sel) = base + cols + (1:cols);
        hist_num(sel) = base + 2*cols + (1:cols);
        pct_num(sel)  = base + 3*cols + (1:cols);
    end
end

function scale_ax(ax, perc)
    pos = get(ax, 'Position');
    new_width = pos(3) * perc;
    pos(1) = pos(1) + (pos(3) - new_width)/2;
    pos(3) = new_width;
    set(ax, 'Position', pos);
end

function shrink_axes_gap(ax, shrink)
% SHRINK_AXES_GAP  Shrinks this axes' box height around its own vertical
% center by SHRINK (e.g. 0.72 keeps 72% of the row's height), leaving a
% real blank gap split evenly above and below. subplot(...,'align') packs
% rows edge-to-edge with only a tiny built-in margin that doesn't scale up
% with how many sub-rows/text elements are packed into the figure -- with 5
% sub-rows per stimulus (image/raster/PSTH/GT-vs-ARM-value-bar) plus the
% image panel's own overflow zscore/mu-std/filename text and every bar
% panel's title (drawn just above its own axes box), that tiny margin isn't
% enough and content bleeds into the neighboring row. Applying this to every
% one of the 5 panels means each row loses a little height but every
% boundary between two rows gets a real gap (half from each side), which is
% what actually stops the bleed -- must be called AFTER a panel's content
% (and any other Position-touching calls like scale_ax) so it isn't undone.
    pos = get(ax, 'Position');
    new_h = pos(4) * shrink;
    pos(2) = pos(2) + (pos(4) - new_h)/2;
    pos(4) = new_h;
    set(ax, 'Position', pos);
end

function compress_to_grid_area(ax, margin_frac)
% COMPRESS_TO_GRID_AREA  Rescales this axes' [x, width] into the left
% (1-margin_frac) fraction of the figure, freeing a right-side margin
% strip for the page-level legend + median_post summary panel (see
% draw_page_margin) without shrinking the per-stimulus grid's actual
% pixel size -- fig_w already grew by the same margin_frac, so this just
% compresses the grid back to the footprint it had before the figure grew.
    pos = get(ax, 'Position');
    pos(1) = pos(1) * (1 - margin_frac);
    pos(3) = pos(3) * (1 - margin_frac);
    set(ax, 'Position', pos);
end

function draw_page_margin(fig, data_table, margin_frac, fontsize_mult, x_jitter)
% DRAW_PAGE_MARGIN  Once per page (not per stimulus, per compress_to_grid_
% area's freed-up right-side strip): a shared legend for plot_arm_gt_value
% _bar's med/mean/zscore/resp line colors (metric_line_colors), plus the
% median_post GT-vs-ARM summary panel -- the SAME picture plot_gt_arm_
% summary_figure draws in its own separate figure (via the shared
% plot_gt_arm_metric_panel), folded into this page too so it's visible
% without opening that other file. Silently skips the summary panel (but
% still draws the legend) when data_table doesn't carry the gt_median_post
% reference column -- a GT-side page, or an arm page whose gt_ref join
% found nothing.
    x0 = 1 - margin_frac + 0.015;
    w  = margin_frac - 0.03;
    % legend and panel are both about the GT-vs-ARM lines: nothing to draw without GT
    if ~ismember('gt_median_post', data_table.Properties.VariableNames), return; end

    % Both moved down from the top edge (legend was crowding it), and the
    % median panel moved down MORE than the legend so there's a real gap
    % between the two instead of the legend's box bleeding into the panel
    % title underneath it.
    leg_ax = axes(fig, 'Position', [x0, 0.78, w, 0.12]);
    hold(leg_ax, 'on');
    c = metric_line_colors();
    plot(leg_ax, NaN, NaN, '-s', 'Color', c.med,    'MarkerFaceColor', c.med,    'DisplayName','med');
    plot(leg_ax, NaN, NaN, '-^', 'Color', c.mean,   'MarkerFaceColor', c.mean,   'DisplayName','mean');
    plot(leg_ax, NaN, NaN, '-d', 'Color', c.zscore, 'MarkerFaceColor', c.zscore, 'DisplayName','zscore');
    plot(leg_ax, NaN, NaN, '-o', 'Color', c.resp,   'MarkerFaceColor', c.resp,   'DisplayName','resp');
    if all(ismember({'p_value_sign','gt_p_value_sign'}, data_table.Properties.VariableNames))
        plot(leg_ax, NaN, NaN, '-v', 'Color', c.sign, 'MarkerFaceColor', c.sign, 'DisplayName','-log10 sign p (scaled)');
    end
    legend(leg_ax, 'Location','north', 'FontSize', 7*fontsize_mult, 'Interpreter','none', 'Box','off');
    axis(leg_ax, 'off');

    if all(ismember({'median_post','gt_median_post'}, data_table.Properties.VariableNames))
        med_ax = axes(fig, 'Position', [x0, 0.28, w, 0.30]);
        plot_gt_arm_metric_panel(med_ax, data_table, 'median_post', 'gt_median_post', ...
            'median post (all stimuli)', '%.3f', fontsize_mult, x_jitter);
    end
end

function scale_factor = plot_img_zscore_class(stim_idx, data_table, grapes, subplot_ax, fontsize_mult, image_cache)
    strength_thr = 5;
    img_idx = data_table.stim_number(stim_idx);

    if ~isempty(image_cache) && isa(image_cache, 'containers.Map') && isKey(image_cache, img_idx)
        A_cached = image_cache(img_idx);
    else
        A_cached = [];
    end

    imagename = fullfile(grapes.ImageNames.folder{img_idx} , grapes.ImageNames.name{img_idx});
    if isempty(A_cached) && ~exist(imagename, "file")
        fprintf("%s not found", imagename);
    end
    if contains(lower(imagename),'.jp')
        img_name_color = 'black';
        [~,fname,~] = fileparts(imagename);
        if length(fname) > 25
            fname = [fname(1:10) '..' fname(end-15:end)];
        end
        if any("selectable" == string(grapes.ImageNames.Properties.VariableNames))
            if grapes.ImageNames.selectable(img_idx) == 1
                img_name_color = 'magenta';
            elseif grapes.ImageNames.selectable(img_idx) == 2
                img_name_color = 'blue';
            elseif grapes.ImageNames.selectable(img_idx) == 3
                img_name_color = 'cyan';
            else
                img_name_color = 'black';
            end
        else
            if ismember('concept_number', data_table.Properties.VariableNames)
                if str2num(data_table.concept_number(stim_idx)) > 1
                    img_name_color = 'magenta';
                else
                    img_name_color = 'black';
                end
            end
        end

        img_name_pos_x = 50;
        img_name_pos_y = 195;
        scale_factor = 1;
        zm_pos_x = 210;
        zm_pos_y = 40;
        zscore_pos_x = 190;
        zscore_pos_y = 110;
        try
            if ~isempty(A_cached)
                A = A_cached;
            else
                A = imread(imagename);
            end
            [~, width, ~] = size(A);
            scale_factor = width/160;

            img_name_pos_x = 50 * scale_factor;
            img_name_pos_y = 195 * scale_factor;
            zm_pos_x = 210 * scale_factor;
            zm_pos_y = 40 * scale_factor;
            zscore_pos_x = 190 * scale_factor;
            zscore_pos_y = 110 * scale_factor;

            image(A); set(subplot_ax,'PlotBoxAspectRatio',[1 1 1]);
            if contains(grapes.ImageNames.folder{img_idx},'RSVPCategLocaliz') || ...
                contains(grapes.ImageNames.folder{img_idx},'categ_pics') || ...
                contains(grapes.ImageNames.folder{img_idx},'categ_localiz')
                colormap(gray(256))
            end
        catch
            img_name_pos_x = 500;
            img_name_pos_y = 0.5;
            set(subplot_ax,'PlotBoxAspectRatio',[2 1 1]);
            xlim([0, 1000]);
        end
        text(img_name_pos_x,img_name_pos_y,fname,'fontsize',4*fontsize_mult, ...
            'HorizontalAlignment','center', 'interpreter','none', 'Color', img_name_color);
        if data_table.zscore(stim_idx) >= 1e10
            zscore_val = Inf;
        else
            zscore_val = data_table.zscore(stim_idx);
        end
        if data_table.zscore(stim_idx)>=strength_thr && data_table.p_test(stim_idx) && data_table.min_spk_test(stim_idx) && data_table.good_lat(stim_idx)
            text(zm_pos_x,zm_pos_y,'Z_m*','fontsize',5*fontsize_mult,'FontWeight','bold');
            text(zscore_pos_x,zscore_pos_y,sprintf('%2.2f',zscore_val),'fontsize',5*fontsize_mult,'FontWeight','bold');
        elseif data_table.zscore(stim_idx)>=strength_thr && data_table.p_test(stim_idx) && data_table.min_spk_test(stim_idx)
            text(zm_pos_x,zm_pos_y,'Z_m','fontsize',5*fontsize_mult,'FontWeight','bold');
            text(zscore_pos_x,zscore_pos_y,sprintf('%2.2f',zscore_val),'fontsize',5*fontsize_mult,'FontWeight','bold');
        else
            text(zm_pos_x,zm_pos_y,'Z_m','fontsize',5*fontsize_mult);
            text(zscore_pos_x,zscore_pos_y-10*scale_factor,sprintf('%2.2f',zscore_val),'fontsize',5*fontsize_mult);
        end
        % Backward-compatible: mu_base/std_base only shown when present on
        % data_table (e.g. test_rescue_grape_metrics.m's existing calls into
        % this same engine don't need to carry these). n_response (this
        % stimulus's own response-window spike count) and median_post (the
        % per-stimulus response statistic that actually feeds zscore -- see
        % codes_emu/tasks_tools/create_responses_data_parallel.m; mu_base/
        % std_base are pooled, session-wide CONSTANTS with no per-stimulus
        % variation at all) ride along on the SAME text line when available,
        % instead of getting their own subplot row -- a whole panel for one
        % number was a waste of vertical space.
        if all(ismember({'mu_base','std_base'}, data_table.Properties.VariableNames))
            % Existing neighbors: zm_pos_y=40, zscore_pos_y=110, img_name_pos_y=195
            % (all *scale_factor). Placed BELOW the filename (not squeezed
            % between zscore and filename) so it's the bottom-most element,
            % clearly separated from both -- the extra letterbox room from
            % the taller fig_h/shrink_axes_gap below gives this room to sit
            % without bleeding into the raster row underneath.
            mu_sd_txt = sprintf('mu=%.3f sd=%.3f', data_table.mu_base(stim_idx), data_table.std_base(stim_idx));
            % Second line: the actual per-stimulus counts, all three
            % computed from the SAME in-window raster ticks (see
            % plot_response_window_grapes.m) so they're directly comparable:
            % n = total pooled count, med = production's own per-trial
            % MEDIAN (feeds zscore), mean = per-trial arithmetic MEAN
            % (computed here, nothing upstream provides it). Kept on its own
            % line rather than appended to mu/sd -- three more numbers on
            % one line was getting cramped.
            stat_txt = '';
            if ismember('n_response', data_table.Properties.VariableNames)
                stat_txt = sprintf('n=%d', data_table.n_response(stim_idx));
            end
            if ismember('median_post', data_table.Properties.VariableNames)
                stat_txt = strtrim(sprintf('%s med=%.1f', stat_txt, data_table.median_post(stim_idx)));
            end
            if ismember('mean_post', data_table.Properties.VariableNames)
                stat_txt = strtrim(sprintf('%s mean=%.3f', stat_txt, data_table.mean_post(stim_idx)));
            end
            if isempty(stat_txt)
                mu_sd_lines = {mu_sd_txt};
            else
                mu_sd_lines = {mu_sd_txt, stat_txt};
            end
            % Third line: this page's own unit's grapes sign-test p for this
            % stimulus (p_value_sign, create_responses_data_parallel) and its
            % rank among the stimuli on this page (rank 1 = smallest p) --
            % GT's on the GT raster, the arm's on the arm raster.
            if ismember('p_value_sign', data_table.Properties.VariableNames)
                mu_sd_lines{end+1} = sprintf('sign p=%.2g (rank %s)', data_table.p_value_sign(stim_idx), ...
                    p_rank_str(data_table.p_value_sign, stim_idx));
            end
            % VerticalAlignment 'top' pins the FIRST line to this y and lets
            % the block only grow downward -- without it, a 2-line block is
            % vertically CENTERED on this anchor by default, pushing the
            % first line up far enough to collide with the filename text
            % directly above (a real overlap this surfaced on the first try).
            % three lines now (mu/sd, counts, sign test): start a little higher
            % and a touch smaller so the block ends above the raster row
            text(img_name_pos_x, img_name_pos_y+18*scale_factor, mu_sd_lines, ...
                'fontsize',3.1*fontsize_mult, 'HorizontalAlignment','center', 'VerticalAlignment','top');
        end
    else
        if exist(imagename,'file')
            [y,Fs] = audioread(imagename);
            line(subplot_ax,(0:1/Fs:(length(y)-1)/Fs)*1000,y(:,1));
            [~,fname,~] = fileparts(imagename);
            text(500,0.5,fname,'fontsize',5*fontsize_mult,'HorizontalAlignment','center','interpreter','none');
        else
            text(500,0.5,lower(imagename),'fontsize',5*fontsize_mult,'HorizontalAlignment','center','interpreter','none');
        end
    end

    if data_table.p_test(stim_idx)
        sign_test = '*';
    else
        sign_test = '';
    end
    image_ylabel = {[num2str(img_idx) sign_test],['#' num2str(data_table.ntrials(stim_idx))], ...
                    [erase( data_table.channel{stim_idx} , 'chan' ) ':' ...
                     erase( data_table.class{stim_idx} , 'class' )]};

    ylabel(image_ylabel,'rotation',0,'fontsize',5*fontsize_mult,...
        'HorizontalAlignment','right','FontWeight','bold','units','normalized','position',[-0.1 -0.2 0]);
end

function add_bundle_label(stim_idx, data_table, grapes, ...
                          priority_chs_ranking, fontsize_mult, ...
                          scale_factor)
    electrode_id = erase(data_table.channel{stim_idx},'chan');
    color = 'black';
    if exist('priority_chs_ranking', 'var') && numel(priority_chs_ranking)
        if sum(priority_chs_ranking == str2num(electrode_id)) == 1
            color = 'red';
        else
            color = 'black';
        end
    end
    try
        chan_rasters = grapes.rasters.(data_table.channel{stim_idx});
        if isfield(chan_rasters, 'details')
            bundle = chan_rasters.details.ch_label;
            bundle = regexprep(bundle, '[^A-Z]', '');
            text(190*scale_factor,160*scale_factor,bundle,'fontsize', 4*fontsize_mult, 'Color', color)
        end
    catch ME
        msg = getReport(ME);
    end
end

function plot_arm_gt_value_bar(stim_idx, data_table, subplot_ax, fontsize_mult)
% PLOT_ARM_GT_VALUE_BAR  4th sub-row: this stimulus's ACTUAL GT vs ARM
% values (not a %% difference -- see plot_gt_arm_summary_figure for the
% across-stimuli trend view), for the four quantities that are actually
% FUNCTIONS OF THIS STIMULUS (see codes_emu/tasks_tools/
% create_responses_data_parallel.m): response count (n_response),
% median_post ("med" -- matches the med=NNN text on the image panel; this is
% a per-trial MEDIAN, the production pipeline's own field, NOT a mean),
% mean_post ("mean" -- a genuine per-trial arithmetic MEAN, computed locally
% since nothing upstream provides one, see plot_response_window_grapes.m),
% and zscore. Only meaningful for an ARM's page (GT has nothing to compare
% against itself) -- silently turns off when data_table doesn't carry the
% gt_* reference columns run_rescue_grape_analysis.m attaches only for arm
% calls (via plot_response_window_grapes.m's gt_ref argument). Every panel
% uses the same fixed per-metric colors (metric_line_colors) -- the shared
% legend explaining them is drawn once per page, in the margin, not
% repeated on all ~20 individual panels.
%
% Drawn as a connected line/marker plot across the {GT, ARM} x-axis, one
% line per metric -- same idiom as plot_stim_comparison_page_n.m's
% plot_line_graph_n (the per-stimulus comparison page's top-middle panel),
% not a bar chart.
    needed = {'n_response','median_post','mean_post','zscore', ...
              'gt_n_response','gt_median_post','gt_mean_post','gt_zscore'};
    if ~all(ismember(needed, data_table.Properties.VariableNames))
        axis(subplot_ax, 'off');
        return;
    end
    arms = categorical({'GT','ARM'}, {'GT','ARM'});
    resp = [data_table.gt_n_response(stim_idx),  data_table.n_response(stim_idx)];
    med  = [data_table.gt_median_post(stim_idx), data_table.median_post(stim_idx)];
    mn   = [data_table.gt_mean_post(stim_idx),   data_table.mean_post(stim_idx)];
    zsc  = [data_table.gt_zscore(stim_idx),      data_table.zscore(stim_idx)];

    % Fixed per-metric colors (metric_line_colors), shared with the
    % page-level legend drawn once in the margin (see
    % loop_plot_responses_BCM_response_window) -- MATLAB's default color
    % order resets independently per yyaxis side and isn't guaranteed to
    % keep med/mean/zscore/resp visually distinct from each other.
    c = metric_line_colors();

    % resp (n_response) routinely sits in the hundreds while med/mean/zscore
    % stay under ~20 -- sharing one axis would flatten those three to
    % near-invisible, so resp gets its own right-hand axis and the other
    % three (a comparable scale to each other) share the left.
    yyaxis(subplot_ax, 'left');
    hold(subplot_ax, 'on');
    plot(subplot_ax, arms, med, '-s', 'Color', c.med,    'MarkerFaceColor', c.med);
    plot(subplot_ax, arms, mn,  '-^', 'Color', c.mean,   'MarkerFaceColor', c.mean);
    plot(subplot_ax, arms, zsc, '-d', 'Color', c.zscore, 'MarkerFaceColor', c.zscore);

    % per-stimulus grapes sign test, GT's own p -> the arm's own p, as
    % -log10(p), scaled by ONE factor for the whole page so its largest value
    % matches the largest med/mean/zscore on the page (raw -log10 p, ~1-5,
    % sat flat at the bottom of an axis that goes to ~20). Same factor on
    % every stimulus, so the green lines still compare across stimuli.
    if all(ismember({'p_value_sign','gt_p_value_sign'}, data_table.Properties.VariableNames))
        lp = @(p) -log10(max(p, 1e-300));
        page_sgn = [lp(data_table.gt_p_value_sign); lp(data_table.p_value_sign)];
        page_left = [data_table.gt_median_post; data_table.median_post; data_table.gt_mean_post; ...
                     data_table.mean_post; data_table.gt_zscore; data_table.zscore];
        k = max(page_left(isfinite(page_left))) / max(page_sgn(isfinite(page_sgn)));
        if ~isfinite(k) || k <= 0, k = 1; end
        sgn = k * [lp(data_table.gt_p_value_sign(stim_idx)), lp(data_table.p_value_sign(stim_idx))];
        plot(subplot_ax, arms, sgn, '-v', 'Color', c.sign, 'MarkerFaceColor', c.sign);
    end

    yyaxis(subplot_ax, 'right');
    hold(subplot_ax, 'on');
    plot(subplot_ax, arms, resp, '-o', 'Color', c.resp, 'MarkerFaceColor', c.resp);
    % yyaxis colors its ticks/axis-line by default (blue left, orange
    % right) regardless of what color the plotted line actually is --
    % force the right axis itself to match resp's line color.
    subplot_ax.YAxis(2).Color = c.resp;

    % No per-panel legend -- repeating it on ~20 tiny subplots was clutter;
    % one shared legend is drawn once per page instead (see
    % loop_plot_responses_BCM_response_window's margin_legend).
    set(subplot_ax, 'FontSize', 3.5*fontsize_mult);
    title(subplot_ax, 'GT vs ARM values', 'FontSize', 4*fontsize_mult, 'Interpreter','none');
    grid(subplot_ax, 'on');
    box(subplot_ax, 'off');
end

function c = metric_line_colors()
% METRIC_LINE_COLORS  Fixed color per line in plot_arm_gt_value_bar's
% med/mean/zscore/resp panel, reused by the page-level margin legend so
% the swatch colors always match what's actually drawn on every panel.
    c.med    = [0.00 0.45 0.74];   % blue
    c.mean   = [0.85 0.33 0.10];   % orange
    c.zscore = [0.49 0.18 0.56];   % purple
    c.resp   = [0.00 0.00 0.00];   % black
    c.sign   = [0.13 0.55 0.13];   % green: -log10(grapes sign-test p)
end

function s = p_rank_str(p_all, i)
% rank of p_all(i) among the finite values of p_all (1 = smallest p), as 'r/N'
    v = isfinite(p_all);
    if ~v(i), s = '-'; return; end
    s = sprintf('%d/%d', nnz(p_all(v) < p_all(i)) + 1, nnz(v));
end

function plot_gt_arm_summary_figure(data_table, output_dir, extra_label_name, fontsize_mult, x_jitter)
% PLOT_GT_ARM_SUMMARY_FIGURE  One extra, SEPARATE figure (not folded into the
% per-stimulus page grid) summarizing this whole channel/class's GT-vs-ARM
% agreement ACROSS every stimulus at once: one subplot per metric, x-axis =
% {GT, ARM}, one thin paired line + a dot at each end per stimulus (a
% "slope"/paired plot -- lets you see the general trend across every
% stimulus at a glance), plus a box+whiskers at each x position spanning
% mean+/-std (not quartiles -- these are small, often skewed per-stimulus
% samples where the mean/std is the quantity that actually matters here,
% e.g. it's what feeds zscore) with the exact mean/std values annotated
% (extra decimal places on mean_post specifically -- with only 1 decimal it
% was hard to tell whether zscore, which is sensitive to small differences
% in the per-trial mean, was actually being computed correctly).
%
% Each panel's title also carries paired, two-sided GT-vs-ARM tests across
% stimuli (paired_gt_arm_tests): sign test and Wilcoxon signed-rank. A
% seventh panel, the per-stimulus sign-test p (p_value_sign), is added when
% GT's column was joined.
%
% Six metrics: n_response/median_post/mean_post/zscore vary genuinely
% per-stimulus; mu_base/std_base are session-wide CONSTANTS (see
% codes_emu/tasks_tools/create_responses_data_parallel.m) so every
% stimulus's line sits at the same two points for those -- included anyway
% since a single GT-vs-ARM offset is still useful to see, it just won't show
% a per-stimulus trend.
%
% Silently does nothing when data_table doesn't carry the gt_* reference
% columns (a GT-side page, or an arm page whose gt_ref join found nothing).
    metrics = struct( ...
        'field',    {'n_response','median_post','mean_post','zscore','mu_base','std_base'}, ...
        'gt_field', {'gt_n_response','gt_median_post','gt_mean_post','gt_zscore','gt_mu_base','gt_std_base'}, ...
        'label',    {'n response','median post','mean post','zscore','mu base','std base'}, ...
        'fmt',      {'%.2f','%.3f','%.3f','%.2f','%.3f','%.3f'});

    if ~all(ismember({metrics.gt_field}, data_table.Properties.VariableNames))
        return;
    end
    % per-stimulus sign-test p (GT's own vs the arm's own), when available
    if all(ismember({'p_value_sign','gt_p_value_sign'}, data_table.Properties.VariableNames))
        metrics(end+1) = struct('field', 'p_value_sign', 'gt_field', 'gt_p_value_sign', ...
            'label', 'sign-test p (per stimulus)', 'fmt', '%.3g');
    end

    ncols = 3; nrows = ceil(numel(metrics)/ncols);
    fig = figure('Visible','off','Color',[1 1 1],'Position',[10 10 1500 380*nrows], ...
        'GraphicsSmoothing','off','Name','GT vs ARM summary');

    for m = 1:numel(metrics)
        ax = subplot(nrows, ncols, m);
        plot_gt_arm_metric_panel(ax, data_table, metrics(m).field, metrics(m).gt_field, ...
            metrics(m).label, metrics(m).fmt, fontsize_mult, x_jitter);
        % paired GT-vs-ARM tests across stimuli, two-sided
        T = paired_gt_arm_tests(data_table.(metrics(m).gt_field), data_table.(metrics(m).field));
        title(ax, {metrics(m).label, ...
            sprintf('sign test p=%.3g  (ARM>GT in %d, ARM<GT in %d)', T.p_sign, T.n_up, T.n_down), ...
            sprintf('signed-rank p=%.3g', T.p_rank)}, 'Interpreter','none', 'FontSize', 6*fontsize_mult);
    end

    sgtitle(fig, {sprintf('%s: GT vs ARM per-stimulus summary', extra_label_name), ...
        'tests: paired across stimuli, two-sided'}, 'Interpreter','none', 'FontSize', 8*fontsize_mult);

    fig_filename = fullfile(output_dir, sprintf('%s_gt_vs_arm_summary.png', extra_label_name));
    try
        print(fig, fig_filename, '-dpng', '-r150');
    catch
        print(fig, '-dpng','-r150', fig_filename);
    end
    close(fig);
end

function T = paired_gt_arm_tests(gt_vals, arm_vals)
% PAIRED_GT_ARM_TESTS  Paired, two-sided tests of ARM vs GT across stimuli:
% sign test (direction only: how many stimuli went up vs down) and Wilcoxon
% signed-rank (also uses the size of each difference). Stimuli with a
% non-finite value on either side are left out; ties don't count as up/down.
    v = isfinite(gt_vals) & isfinite(arm_vals);
    g = gt_vals(v); a = arm_vals(v);
    d = a - g;
    T.n_up = nnz(d > 0); T.n_down = nnz(d < 0); T.n = numel(d);
    T.p_sign = NaN; T.p_rank = NaN;
    if T.n_up + T.n_down == 0
        T.p_sign = 1; T.p_rank = 1;
        return;
    end
    T.p_sign = signtest(a, g);
    T.p_rank = signrank(a, g);
end

function plot_sign_test_figure(data_table, output_dir, extra_label_name, fontsize_mult)
% PLOT_SIGN_TEST_FIGURE  Separate page: every stimulus's sign-test p
% (p_value_sign -- baseline vs post spike counts per trial, one-sided, the
% test behind the '*' on the raster pages), GT and ARM side by side as
% -log10(p), with the p = 0.05 line. Silently skipped without the GT column.
    if ~all(ismember({'p_value_sign','gt_p_value_sign','stim_number'}, data_table.Properties.VariableNames))
        return;
    end
    pg = data_table.gt_p_value_sign; pa = data_table.p_value_sign;
    n = height(data_table);
    lg = -log10(max(pg, 1e-300)); la = -log10(max(pa, 1e-300));
    fig = figure('Visible','off','Color',[1 1 1],'Position',[10 10 max(700, 60*n + 200) 500], ...
        'GraphicsSmoothing','off','Name','sign test per stimulus');
    ax = axes(fig); hold(ax, 'on');
    hb = bar(ax, 1:n, [lg la], 'grouped');
    hb(1).FaceColor = [0.15 0.15 0.15]; hb(2).FaceColor = [0.75 0.1 0.1];
    yline(ax, -log10(0.05), 'k--', 'p = 0.05', 'LabelHorizontalAlignment', 'left', 'FontSize', 6*fontsize_mult);
    set(ax, 'XTick', 1:n, 'XTickLabel', arrayfun(@(s) sprintf('%d', s), data_table.stim_number, 'UniformOutput', false), ...
        'XTickLabelRotation', 90, 'FontSize', 6*fontsize_mult);
    xlabel(ax, 'stimulus'); ylabel(ax, '-log_{10}(sign-test p)');
    legend(ax, {sprintf('GT  (%d/%d p<=0.05)', nnz(pg <= 0.05), nnz(isfinite(pg))), ...
                sprintf('ARM (%d/%d p<=0.05)', nnz(pa <= 0.05), nnz(isfinite(pa)))}, ...
        'Location', 'northeast', 'FontSize', 6*fontsize_mult, 'Box', 'off');
    title(ax, sprintf('%s: per-stimulus sign test (baseline vs post, one-sided)', extra_label_name), ...
        'Interpreter', 'none', 'FontSize', 7*fontsize_mult);
    grid(ax, 'on'); box(ax, 'off');
    fig_filename = fullfile(output_dir, sprintf('%s_sign_test.png', extra_label_name));
    try
        print(fig, fig_filename, '-dpng', '-r150');
    catch
        print(fig, '-dpng','-r150', fig_filename);
    end
    close(fig);
end

function plot_gt_arm_metric_panel(ax, data_table, field, gt_field, label, fmt, fontsize_mult, x_jitter)
% PLOT_GT_ARM_METRIC_PANEL  One GT-vs-ARM paired-line + mean/std-box panel
% for a single metric, across every stimulus in data_table -- the shared
% body behind plot_gt_arm_summary_figure's per-metric loop AND the single
% median_post panel folded into the main raster page's side margin (see
% loop_plot_responses_BCM_response_window), so both places draw the exact
% same picture instead of two hand-kept copies.
    gt_vals  = data_table.(gt_field);
    arm_vals = data_table.(field);
    valid = isfinite(gt_vals) & isfinite(arm_vals);
    gt_vals = gt_vals(valid); arm_vals = arm_vals(valid);

    % Optional x jitter, one offset per stimulus shared by both ends so each
    % paired line stays traceable. Fixed-seed local stream: the same page
    % always draws the same way, and the global rng is left untouched.
    if nargin < 8 || isempty(x_jitter), x_jitter = 0; end
    if x_jitter > 0
        rs = RandStream('mt19937ar', 'Seed', 0);
        dx = (2*rand(rs, numel(gt_vals), 1) - 1) * x_jitter;
    else
        dx = zeros(numel(gt_vals), 1);
    end
    x_gt = 1 + dx; x_arm = 2 + dx;

    hold(ax,'on');
    for k = 1:numel(gt_vals)
        hl = plot(ax, [x_gt(k) x_arm(k)], [gt_vals(k) arm_vals(k)], '-', 'Color', [0.6 0.6 0.6], 'LineWidth', 0.5);
        hl.Color(4) = 0.4;
    end
    scatter(ax, x_gt,  gt_vals,  10, [0.15 0.15 0.15], 'filled');
    scatter(ax, x_arm, arm_vals, 10, [0.75 0.1 0.1], 'filled');

    draw_mean_std_box(ax, 1, gt_vals,  [0.15 0.15 0.15], fmt, fontsize_mult);
    draw_mean_std_box(ax, 2, arm_vals, [0.75 0.1 0.1],   fmt, fontsize_mult);

    set(ax, 'XLim', [0.5 2.5], 'XTick', [1 2], 'XTickLabel', {'GT','ARM'}, 'FontSize', 6*fontsize_mult);
    % headroom above the data so the mean/std labels never run into the title
    yl = ylim(ax); yr = max(diff(yl), eps);
    ylim(ax, [yl(1), yl(2) + 0.2*yr]);
    title(ax, label, 'Interpreter','none', 'FontSize', 7*fontsize_mult);
    box(ax, 'off');
end

function draw_mean_std_box(ax, x, vals, color, fmt, fontsize_mult)
% DRAW_MEAN_STD_BOX  A box+whiskers centered on MEAN +/- STD (deliberately
% not the usual quartile boxplot -- see plot_gt_arm_summary_figure), plus a
% text annotation giving the exact numbers at the precision requested for
% each metric (metrics.fmt).
    if isempty(vals), return; end
    mu = mean(vals,'omitnan'); sd = std(vals,'omitnan');
    box_w = 0.15;
    patch(ax, x+box_w*[-1 1 1 -1], [mu-sd mu-sd mu+sd mu+sd], color, ...
          'FaceAlpha', 0.15, 'EdgeColor', color, 'LineWidth', 1.2);
    line(ax, [x x], [mu-sd mu+sd], 'Color', color, 'LineWidth', 1.2);
    line(ax, x+box_w*[-1 1], [mu mu], 'Color', color, 'LineWidth', 1.8);
    txt = sprintf(['mu=' fmt ' sd=' fmt], mu, sd);
    ylims = ylim(ax);
    yoff = 0.04*diff(ylims);
    text(ax, x, mu+sd+yoff, txt, 'FontSize', 4.5*fontsize_mult, 'Color', color, ...
        'HorizontalAlignment','center', 'VerticalAlignment','bottom');
end

function plot_rasters(stim_idx, data_table, grapes, rank_config, subplot_ax, b_yellow_patch, b_short_win, clip_to_window)
    LineFormat.Color = 'blue';
    LineFormat.LineWidth = 0.35;

    if nargin < 8, clip_to_window = false; end
    if b_short_win
        time_pre_ms = grapes.time_pre / 2;
        time_pos_ms = grapes.time_pos / 2;
    else
        time_pre_ms = grapes.time_pre;
        time_pos_ms = grapes.time_pos;
    end

    hold on;
    img_idx = data_table.stim_number(stim_idx);
    spikes1 = grapes.rasters.(data_table.channel{stim_idx}).(data_table.class{stim_idx}).stim{img_idx};

    if iscell(spikes1)
        trials = spikes1;
    else
        trials = num2cell(spikes1,2);
    end
    % clip_to_window: restrict each trial's displayed ticks to [win_lo,win_hi]
    % -- otherwise every panel shows the full raw +/-1000/2000ms window,
    % which for a sparse exclusive spike set can pull in nearby spikes that
    % were actually classified to a DIFFERENT trial, making it look like
    % spikes fall outside the green in/out lines when they're really just
    % passing through this trial's wide display window. Off by default
    % (the regular per-arm rasters should keep showing full context).
    if clip_to_window && ismember('win_lo', data_table.Properties.VariableNames)
        lo = data_table.win_lo(stim_idx); hi = data_table.win_hi(stim_idx);
        trials = cellfun(@(v) v(v>=lo & v<=hi), trials, 'UniformOutput', false);
    end

    if numel(trials)>0
        lst = numel(trials);
        if ~all(cellfun(@isempty,trials))
            plotSpikeRaster(trials,'PlotType','vertline','LineFormat',LineFormat);
        end
        set(subplot_ax,'YLim',[0.5 lst+0.5])
    end

    set(subplot_ax,'XLim',[-time_pre_ms time_pos_ms]);
    limy=ylim;
    if b_yellow_patch
        xpoints = [data_table.onset(stim_idx) data_table.onset(stim_idx) + rank_config.win_length ...
                   data_table.onset(stim_idx) + rank_config.win_length data_table.onset(stim_idx)];
        ypoints = [limy(1) limy(1) limy(2) limy(2)];
        p=patch(xpoints,ypoints,'y');
        set(p,'FaceAlpha',0.2,'linestyle','none');
        uistack(p, 'bottom');
    end
    set(subplot_ax,'FontSize',7);
    axis off
    line([data_table.onset(stim_idx) data_table.onset(stim_idx)],ylim,'linestyle',':','color','r','linewidth',0.5)

    % ---- NEW: our own in/out classification boundary, two solid green lines ----
    if ismember('win_lo', data_table.Properties.VariableNames)
        line([data_table.win_lo(stim_idx) data_table.win_lo(stim_idx)],ylim,'linestyle','-','color',[0 0.6 0],'linewidth',1)
        line([data_table.win_hi(stim_idx) data_table.win_hi(stim_idx)],ylim,'linestyle','-','color',[0 0.6 0],'linewidth',1)
    end
end

function plot_ifr(stim_idx, data_table, grapes, ifr_x, ...
                  ifr_y_max, subplot_ax, fontsize_mult, b_short_win)
    if b_short_win
        time_pre_ms = grapes.time_pre / 2;
        time_pos_ms = grapes.time_pos / 2;
    else
        time_pre_ms = grapes.time_pre;
        time_pos_ms = grapes.time_pos;
    end
    hold on;
    if isnan(data_table.IFR{stim_idx}(1)) && all(isnan(data_table.IFR{stim_idx}))
        IFR_sm = data_table.IFR{stim_idx}';
    else
        smooth_bin = ceil(50 * (ifr_x(2)-ifr_x(1)));
        IFR_sm = smooth(data_table.IFR{stim_idx},smooth_bin);
    end

    lo=line(subplot_ax,ifr_x,IFR_sm,'Color','r','Linewidth',1);

    if ifr_y_max == -1
        limy=ylim;
        ifr_y_max = (limy(2));
    end
    xlim([-time_pre_ms time_pos_ms])

    if ~data_table.good_lat(stim_idx)
        title(sprintf('lat=%3.0f dur=%3.0f',data_table.onset(stim_idx),data_table.dura(stim_idx)),'fontsize',5*fontsize_mult,'interpreter','none','VerticalAlignment','middle');
    else
        title(sprintf('lat=%3.0f dur=%3.0f',data_table.onset(stim_idx),data_table.dura(stim_idx)),'fontsize',5*fontsize_mult,'FontWeight','bold','interpreter','none','VerticalAlignment','middle');
    end
    box off
    line(xlim,[data_table.ifr_thr(stim_idx) data_table.ifr_thr(stim_idx)],'linestyle','--','color','k')

    if ifr_y_max <= 0
        ifr_y_max = 1;
    end
    set(subplot_ax,'YLim',[0 ifr_y_max]);
    line([0 0],[0 ifr_y_max],'linestyle',':')
    if isfield(grapes,'ISI_min')
        for kk=1:floor(grapes.time_pos/(grapes.ISI_min*1000))
            line(1000*[grapes.ISI_min*kk grapes.ISI_min*kk],[0 ifr_y_max],'linestyle',':','color','k')
        end
        line(1000*[-grapes.ISI_min -grapes.ISI_min],[0 ifr_y_max],'linestyle',':','color','k')
    else
        line([500 500],[0 ifr_y_max],'linestyle',':','color','k')
        line([1000 1000],[0 ifr_y_max],'linestyle',':','color','k')
    end
    line([data_table.onset(stim_idx) data_table.onset(stim_idx)],[0 ifr_y_max],'linestyle','-.','color','b','linewidth',0.5)

    limy=ylim;
    xpoints = [data_table.onset(stim_idx) data_table.onset(stim_idx)+data_table.dura(stim_idx) data_table.onset(stim_idx)+data_table.dura(stim_idx) data_table.onset(stim_idx)];
    ypoints = [limy(1) limy(1) limy(2) limy(2)];
    p=patch(xpoints,ypoints,'c');
    set(p,'FaceAlpha',0.2,'linestyle','none');
    uistack(p, 'bottom');

    % ---- NEW: our own in/out classification boundary, two solid green lines ----
    if ismember('win_lo', data_table.Properties.VariableNames)
        lg1 = line([data_table.win_lo(stim_idx) data_table.win_lo(stim_idx)],[0 ifr_y_max],'linestyle','-','color',[0 0.6 0],'linewidth',1.2);
        lg2 = line([data_table.win_hi(stim_idx) data_table.win_hi(stim_idx)],[0 ifr_y_max],'linestyle','-','color',[0 0.6 0],'linewidth',1.2);
        uistack([lg1 lg2], 'top');
    end

    uistack(lo, 'top');
end

function add_figure_title(grapes, rank_config, ss_num, fig_num, extra_label)
% Split across 2 lines -- a single line with everything (label + exp_type +
% subscr/fig + resp_win + legend note) ran past the figure width and was
% clipped off on BOTH edges instead of wrapping.
    line1 = sprintf('%s, %s responses (subscr:%d fig:%d)', extra_label, grapes.exp_type, ss_num, fig_num);
    line2 = sprintf('resp_win: %dms cent dura (ntsd=%d) -- GREEN lines = in/out boundary', ...
        rank_config.win_length, rank_config.nstd);
    sgtitle({line1, line2}, 'interpreter','none', 'fontsize', 8);
end
