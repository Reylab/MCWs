function f = temperature_diagram(tim_dir, nm, clu_par, out_png, ax)
% TEMPERATURE_DIAGRAM  The wave_clus temperature diagram (same drawing as
% wave_clus's mark_clusters_temperature_diagram.m / sort_and_plot_spikes.m):
% size of every SPC cluster vs temperature, the min_clus line, the selected
% temperature, x marks on the clusters find_temp considered, grey shading
% past the elbow, and one coloured dot per final class at the temperature /
% SPC cluster it was taken from.
%
% Reads data_<nm>.dg_01 (SPC tree) and times_<nm>.mat (par, Temp,
% gui_status, cluster_class, forced) from tim_dir.
%
%   temperature_diagram(tim_dir, nm, clu_par, out_png)      save a PNG
%   f = temperature_diagram(tim_dir, nm, clu_par, '')       return the figure
%   ok = temperature_diagram(tim_dir, nm, clu_par, '', ax)  draw into axes ax
%
% clu_par: fallback mintemp/maxtemp/tempstep/min_clus when the times file has
% no par. Returns [] (or false with ax) when there is no .dg_01.
    if nargin < 4, out_png = ''; end
    if nargin < 5, ax = []; end
    into_ax = ~isempty(ax);
    f = []; if into_ax, f = false; end

    dg = fullfile(tim_dir, ['data_' nm '.dg_01']);
    if ~exist(dg, 'file')
        d = dir(fullfile(tim_dir, 'data_*.dg_01'));
        if isempty(d), return; end
        dg = fullfile(tim_dir, d(1).name);
    end
    tree = load(dg);
    if isempty(tree) || size(tree, 2) < 5, return; end

    par = struct('mintemp', 0, 'maxtemp', 0.251, 'tempstep', 0.01, 'min_clus', 20, 'temp_plot', 'log');
    if nargin >= 3 && isstruct(clu_par)
        for fld = fieldnames(clu_par)', par.(fld{1}) = clu_par.(fld{1}); end
    end
    G = struct();
    tf = fullfile(tim_dir, ['times_' nm '.mat']);
    if exist(tf, 'file')
        G = load(tf, 'par', 'Temp', 'gui_status', 'cluster_class', 'forced');
        if isfield(G, 'par')
            for fld = {'mintemp', 'maxtemp', 'tempstep', 'min_clus', 'temp_plot'}
                if isfield(G.par, fld{1}) && ~isempty(G.par.(fld{1})), par.(fld{1}) = G.par.(fld{1}); end
            end
        end
    end
    mintemp = double(par.mintemp); maxtemp = double(par.maxtemp);
    tstep = double(par.tempstep);  min_clus = double(par.min_clus);

    num_temp = min(size(tree, 1), floor((maxtemp - mintemp) / tstep));
    if num_temp < 2, return; end
    temps = mintemp + (1:num_temp) * tstep;
    sizes = tree(1:num_temp, 5:end);

    current_temp = NaN; auto_sort = [];
    if isfield(G, 'gui_status')
        if isfield(G.gui_status, 'current_temp'), current_temp = double(G.gui_status.current_temp); end
        if isfield(G.gui_status, 'auto_sort_info'), auto_sort = G.gui_status.auto_sort_info; end
    end

    if ~into_ax
        f = figure('Visible', 'off', 'Position', [100 100 900 600], 'Color', 'w');
        ax = axes(f);
    end
    hold(ax, 'on');
    is_log = ~strcmpi(char(par.temp_plot), 'lin');
    if is_log, set(ax, 'YScale', 'log'); end
    ytop = max(sizes(:)) * 1.5;

    % grey: past the elbow (find_temp ignores these temperatures)
    if isstruct(auto_sort) && isfield(auto_sort, 'elbow') && ~isempty(auto_sort.elbow)
        x0 = mintemp + tstep * double(auto_sort.elbow);
        x1 = mintemp + tstep * num_temp;
        if x0 < x1
            patch(ax, [x0 x1 x1 x0], [1 1 ytop ytop], [0.9 0.9 0.9], 'EdgeColor', 'none');
        end
    end
    plot(ax, [mintemp, maxtemp - tstep], [min_clus min_clus], 'k:');
    plot(ax, temps, sizes);
    if isfinite(current_temp)
        t_sel = mintemp + current_temp * tstep;
        plot(ax, [t_sel t_sel], [1 tree(1, 5)], 'k:');
    end
    % x: clusters find_temp considered
    if isstruct(auto_sort) && isfield(auto_sort, 'peaks') && ~isempty(auto_sort.peaks)
        [xp, yp] = find(auto_sort.peaks);
        keep = xp <= size(tree, 1) & yp + 4 <= size(tree, 2);
        xp = xp(keep); yp = yp(keep);
        plot(ax, mintemp + xp * tstep, tree(sub2ind(size(tree), xp, yp + 4)), 'xk', 'MarkerSize', 7, 'LineWidth', 0.9);
    end
    % one dot per final class, wave_clus colours
    colors = [0 0 1; 1 0 0; 0 0.5 0; 0.620690 0 0; 0.413793 0 0.758621; 0.965517 0.517241 0.034483; ...
              0.448276 0.379310 0.241379; 1 0.103448 0.724138; 0.545 0.545 0.545; 0.586207 0.827586 0.310345; ...
              0.965517 0.620690 0.862069; 0.620690 0.758621 1];
    if isfield(G, 'Temp') && isfield(G, 'cluster_class') && isfield(G, 'gui_status') ...
            && isfield(G.gui_status, 'original_classes')
        classes = G.cluster_class(:, 1);
        oc = double(G.gui_status.original_classes(:));
        Tsel = double(G.Temp(:));
        if numel(oc) == numel(classes)
            not_forced = true(size(classes));
            if isfield(G, 'forced') && numel(G.forced) == numel(classes), not_forced = ~logical(G.forced(:)); end
            for i = 1:min(max(classes), numel(Tsel))
                o = oc(classes == i & not_forced & oc > 0);
                if isempty(o), o = 1; end          % null original cluster: plot as cluster 1 (wave_clus)
                o = mode(o);
                if Tsel(i) < 1 || Tsel(i) > size(tree, 1) || 4 + o > size(tree, 2), continue; end
                plot(ax, mintemp + Tsel(i) * tstep, tree(Tsel(i), 4 + o), '.', ...
                    'Color', colors(mod(i - 1, size(colors, 1)) + 1, :), 'MarkerSize', 20);
            end
        end
    end
    xlim(ax, [mintemp maxtemp]);
    if is_log, ylim(ax, [1 ytop]); end
    xlabel(ax, 'Temperature'); ylabel(ax, 'Clusters size');
    box(ax, 'off');

    if into_ax
        f = true;
        return;
    end
    title(ax, sprintf('%s   temperature diagram', nm), 'Interpreter', 'none');
    if ~isempty(out_png)
        print(f, out_png, '-dpng', '-r130');
        close(f);
        f = [];
    end
end
