function pvw_matlab(step)
% PVW_MATLAB  MATLAB steps of the python-vs-wave_clus analysis (see README.md).
% Every step reads channels.json (one entry per GT unit) and is read-only on
% the sorts; outputs go to the validation folders and to out_dir.
%
%   pvw_matlab('gt_export')    GT cluster_class, spikes, forced per channel -> out_dir/gt/gt_<ch>.mat (for the python steps)
%   pvw_matlab('wc_export')    each unit's wave_clus run: cluster_class, forced, spikes -> out_dir/wc/<run>_ch<ch>.mat
%   pvw_matlab('gt_vs_python') GT vs newest python only
%   pvw_matlab('wave_clus')    per wave_clus run in channels.json: GT vs wave_clus, and wave_clus vs
%                              python with python as the arm (its GT-matched class)
%   pvw_matlab('validations')  both of the above: GT vs newest python, and per wave_clus run GT vs wave_clus +
%                              wave_clus vs python (GT-closest clusters on every side); skips finished runs
%   pvw_matlab('diagnostics')  python_loss_figures per unit
%   pvw_matlab('tables')       out_dir/par_compare.csv, wc_python_match.csv, raster_metrics.csv
%   pvw_matlab('all')          all of the above, in that order

here = fileparts(mfilename('fullpath'));
V = fileparts(here);
addpath(V, fullfile(fileparts(V), 'real_response_analysis'), fullfile(fileparts(V), 'responses'));
cfgname = getenv('PVW_CONFIG'); if isempty(cfgname), cfgname = 'channels.json'; end   % channels_tuned.json for the tuned trial
C = jsondecode(fileread(fullfile(here, cfgname)));
U = C.units;
if ~exist(C.out_dir, 'dir'), mkdir(C.out_dir); end
steps = {'gt_export', 'wc_export', 'validations', 'diagnostics', 'tables'};
if ~strcmp(step, 'all'), steps = {step}; end
for s = steps
    fprintf('\n===== pvw_matlab: %s =====\n', s{1});
    feval(['step_' s{1}], C, U);
end
end

% -------------------------------------------------------------------------
function step_gt_export(C, U)
    d = fullfile(C.out_dir, 'gt'); if ~exist(d, 'dir'), mkdir(d); end
    for ch = unique([U.ch])
        u = U(find([U.ch] == ch, 1));
        G = load(fullfile(C.single_channel, u.folder, 'hand selected', u.times_file), 'cluster_class', 'spikes', 'forced');
        cluster_class = G.cluster_class; spikes = single(G.spikes); forced = logical(G.forced(:)); %#ok<NASGU>
        save(fullfile(d, sprintf('gt_%d.mat', ch)), 'cluster_class', 'spikes', 'forced', '-v7');
    end
end

function step_wc_export(C, U)
% plain -v7 copies of the wave_clus runs' labels / forced flags / waveforms for the python figures
    d = fullfile(C.out_dir, 'wc'); if ~exist(d, 'dir'), mkdir(d); end
    for u = U'
        for w = cellstr(u.wave_clus)'
            f = fullfile(d, sprintf('%s_ch%d.mat', w{1}, u.ch));
            if exist(f, 'file'), continue; end
            W = load(fullfile(C.single_channel, u.folder, w{1}, u.times_file), 'cluster_class', 'forced', 'spikes');
            cluster_class = W.cluster_class; forced = logical(W.forced(:)); spikes = single(W.spikes); %#ok<NASGU>
            save(f, 'cluster_class', 'forced', 'spikes', '-v7');
        end
    end
end

function step_validations(C, U)
    step_gt_vs_python(C, U);
    step_wave_clus(C, U);
end

function step_gt_vs_python(C, U)
    for u = U'
        root = fullfile(C.single_channel, u.folder);
        out = fullfile(root, ['validation_gt_vs_' u.python_arm u.tok u.py_tag]);
        if ~done(out)
            c = validation_sessions(u.session);
            c.arms = struct('name', 'python_sd1', 'dir', fullfile(root, u.python_arm), 'times_file', u.times_file, ...
                'grapes_file', 'grapes_blanks_rescue.mat', 'class', pyclass(u));
            c.out_dir = out; c.venn_detections = true; c.channel_subsets = true;
            run_validation(c);
        end
    end
end

function step_wave_clus(C, U)
    for u = U'
        root = fullfile(C.single_channel, u.folder);
        todo = {};
        for w = cellstr(u.wave_clus)'
            a = fullfile(root, w{1}, ['validation_gt_vs_' w{1} u.tok]);
            b = fullfile(root, w{1}, sprintf('validation_%s_vs_python_sd1%s%s%s', w{1}, u.tok, u.py_tag, u.wc_tag));
            a2 = fullfile(root, ['validation_gt_vs_' w{1} u.tok]);          % a finished channel-level copy counts
            if ~((done(a) || done(a2)) && done(b)), todo{end+1} = w{1}; end %#ok<AGROW>
        end
        if ~isempty(todo)
            run_wave_clus_validations(u.session, todo, 'python', u.python_arm, 'redo_gt', false, ...
                'python_class', pyclass(u), 'py_tag', [u.py_tag u.wc_tag], 'wc_class', wcclass(u));
        end
    end
end

function step_diagnostics(C, U)
    for u = U'
        root = fullfile(C.single_channel, u.folder);
        python_loss_figures(u.session, 'python', u.python_arm, 'before', u.python_before, 'python_class', pyclass(u), ...
            'out_dir', fullfile(root, ['validation_gt_vs_' u.python_arm u.tok u.py_tag], 'diagnostics'));
    end
end

function step_tables(C, U)
    % shared parameters: wave_clus par vs python par
    rows = {};
    for f = unique({U.folder})
        u = U(find(strcmp({U.folder}, f{1}), 1)); root = fullfile(C.single_channel, f{1});
        pp = flat(getfield(load(fullfile(root, u.python_arm, u.times_file), 'par'), 'par'), ''); %#ok<GFLD>
        for w = cellstr(u.wave_clus)'
            wf = fullfile(root, w{1}, u.times_file); if ~exist(wf, 'file'), continue; end
            wp = flat(getfield(load(wf, 'par'), 'par'), ''); %#ok<GFLD>
            for n = intersect(fieldnames(wp), fieldnames(pp))'
                rows(end+1, :) = {f{1}, w{1}, n{1}, show(wp.(n{1})), show(pp.(n{1})), same(wp.(n{1}), pp.(n{1}))}; %#ok<AGROW>
            end
        end
    end
    writetable(cell2table(rows, 'VariableNames', {'channel','run','param','wave_clus','python','same'}), ...
        fullfile(C.out_dir, 'par_compare.csv'));

    % each wave_clus run's GT cluster vs python's GT cluster (fair = both ways >= fair_min)
    % and the raster numbers per sort
    rows = {}; rr = {};
    for u = U'
        root = fullfile(C.single_channel, u.folder);
        G = load(fullfile(root, 'hand selected', u.times_file), 'cluster_class');
        gt = sort(G.cluster_class(G.cluster_class(:, 1) == u.gt_class, 2));
        P = load(fullfile(root, u.python_arm, u.times_file), 'cluster_class');
        pc = pyclass(u); if isempty(pc), pc = best(P.cluster_class, gt); end
        pt = sort(P.cluster_class(P.cluster_class(:, 1) == pc, 2));
        pv = fullfile(root, ['validation_gt_vs_' u.python_arm u.tok u.py_tag], 'stimulus_rasters');
        rr = addr(rr, u, 'GT', fullfile(pv, 'gt', 'data_table.mat'));
        rr = addr(rr, u, 'python', fullfile(pv, 'python_sd1', 'data_table.mat'));
        for w = cellstr(u.wave_clus)'
            f = fullfile(root, w{1}, u.times_file); if ~exist(f, 'file'), continue; end
            W = load(f, 'cluster_class');
            wc = wcclass(u); if isempty(wc), wc = best(W.cluster_class, gt); end
            wt = sort(W.cluster_class(W.cluster_class(:, 1) == wc, 2));
            [rec, prec] = overlap(wt, pt, 0.5);      % python's GT cluster vs wave_clus's GT cluster
            rows(end+1, :) = {u.id, u.ch, u.gt_class, w{1}, wc, numel(wt), pc, numel(pt), rec, prec, ...
                rec >= C.fair_min && prec >= C.fair_min}; %#ok<AGROW>
            % the finished GT-vs-wave_clus run: inside the wave_clus folder, else at the channel level
            if isempty(wcclass(u))
                gv = fullfile(root, w{1}, ['validation_gt_vs_' w{1} u.tok]);
                if ~done(gv), gv = fullfile(root, ['validation_gt_vs_' w{1} u.tok]); end
                rr = addr(rr, u, w{1}, fullfile(gv, 'stimulus_rasters', w{1}, 'data_table.mat'));
            else   % fixed wave_clus class: its rasters come from the wave_clus-vs-python run (it is the reference there)
                gv = fullfile(root, w{1}, sprintf('validation_%s_vs_python_sd1%s%s%s', w{1}, u.tok, u.py_tag, u.wc_tag));
                rr = addr(rr, u, sprintf('%s c%d', w{1}, wc), fullfile(gv, 'stimulus_rasters', w{1}, 'data_table.mat'));
            end
        end
    end
    writetable(cell2table(rows, 'VariableNames', {'unit','ch','gt_class','wave_clus','wc_class','wc_size', ...
        'python_class','python_size','recall','precision','fair'}), fullfile(C.out_dir, 'wc_python_match.csv'));
    writetable(cell2table(rr, 'VariableNames', {'unit','ch','gt_class','sort','n_stim','mean_zscore','mean_n_response', ...
        'mean_median_post','mu_base','std_base','n_responsive','median_sign_p','cluster_total'}), ...
        fullfile(C.out_dir, 'raster_metrics.csv'));
end

% ------------------------------------------------------------------ helpers
function y = done(d), y = exist(fullfile(d, 'pdf_reports', 'validation_report.pdf'), 'file') == 2; end

function c = wcclass(u)
% the wave_clus class fixed for this unit in channels.json ([] = matched to GT)
    c = []; if isfield(u, 'wc_class') && ~isempty(u.wc_class), c = u.wc_class; end
end

function c = pyclass(u)
% the python class fixed for this unit in channels.json ([] = match to GT)
    c = []; if isfield(u, 'python_class') && ~isempty(u.python_class), c = u.python_class; end
end

function rr = addr(rr, u, name, f)
    if ~exist(f, 'file'), return; end
    D = load(f); D = D.data_table;
    ct = NaN; if ismember('cluster_total', D.Properties.VariableNames), ct = D.cluster_total(1); end
    rr(end+1, :) = {u.id, u.ch, u.gt_class, name, height(D), mean(D.zscore, 'omitnan'), mean(D.n_response, 'omitnan'), ...
        mean(D.median_post, 'omitnan'), D.mu_base(1), D.std_base(1), nnz(D.p_test & D.zscore >= 3), ...
        median(D.p_value_sign, 'omitnan'), ct};
end

function c = best(cc, ref)
% cluster of cc holding the most spikes of ref (0.5 ms)
    t = cc(:, 2); lab = cc(:, 1); [ts, o] = sort(t);
    k = o(interp1(ts, 1:numel(ts), ref, 'nearest', 'extrap'));
    L = lab(k(abs(t(k) - ref) <= 0.5));
    c = mode(L(L > 0));
end

function [rec, prec] = overlap(a, b, tol)
% share of a with a spike of b within tol (recall), and share of b with one of a (precision)
    k = interp1(b, 1:numel(b), a, 'nearest', 'extrap'); rec = mean(abs(b(k) - a) <= tol);
    k = interp1(a, 1:numel(a), b, 'nearest', 'extrap'); prec = mean(abs(a(k) - b) <= tol);
end

function s = flat(x, pre)
    s = struct();
    if ~isstruct(x), return; end
    for f = fieldnames(x)'
        v = x.(f{1}); nm = matlab.lang.makeValidName([pre f{1}]);
        if isstruct(v) && isscalar(v)
            t = flat(v, [pre f{1} '_']); for g = fieldnames(t)', s.(g{1}) = t.(g{1}); end
        else
            s.(nm) = v;
        end
    end
end

function t = show(v)
    if ischar(v) || isstring(v), t = char(v);
    elseif isnumeric(v) || islogical(v)
        if numel(v) <= 6, t = mat2str(double(v), 4); else, t = sprintf('[%dx%d %s]', size(v, 1), size(v, 2), class(v)); end
    elseif iscell(v), t = sprintf('{cell %dx%d}', size(v, 1), size(v, 2));
    else, t = class(v); end
end

function y = same(a, b)
    if (isnumeric(a) || islogical(a)) && (isnumeric(b) || islogical(b))
        y = isequal(size(a), size(b)) && all(abs(double(a(:)) - double(b(:))) <= 1e-9 * max(1, abs(double(a(:)))));
    else
        y = isequal(show(a), show(b));
    end
end
