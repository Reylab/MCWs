function plot_subset_rasters(subset_names, varargin)
% PLOT_SUBSET_RASTERS  Stimulus raster pages for saved spike subsets
% (save_validation_subsets), same layout as the run's raster pages. No
% GT-vs-ARM comparison panels or summary page: a subset is one piece of the
% GT / arm overlap (or of what neither kept), so there is no counterpart to
% compare it against.
%
%   cd <validation out_dir or any folder under it>
%   plot_subset_rasters({'gt_only', 'arm_only'})
%   plot_subset_rasters('default')   % gt_only, arm_only, matched, q_not_rescued
%                                    % and every qnr_<property> (run_validation's set)
%   plot_subset_rasters('all')                          % every saved subset
%   plot_subset_rasters('matched', 'arm', 'python_sd2') % arm by name or index
%   plot_subset_rasters([], 'times', t_ms, 'name', 'weird_spikes')  % any spike times
%
% Name-value:
%   'dir'    validation out_dir (or a folder under it), default pwd
%   'arm'    which arm's subsets file, name or index, default 1
%   'times'  custom spike times (ms) instead of saved subsets; needs 'name'
%   'name'   label for 'times'
%   'side'   for 'times': 'gt' = the reference unit's latency / windows, 'arm' (default) = the arm's
%   'clip_to_window'  only draw ticks inside each stimulus's window, default false
%   'out_dir'  write each subset's pages to <out_dir>/<name> (default: the run's subset_rasters/<arm>/)
%   'window' 'defined' (default) = the full run's latency / duration windows (ref_tables); 'default' = the
%            run's default-window version (ref_tables_default: late-latency / long-duration stimuli get
%            grapes' default post window). The page footer says which.
%
% Apples to apples with the run's own raster pages: every subset takes its
% source unit's latency, duration and post window per stimulus (gt_ref for
% GT-side subsets, the arm's table -- arm1_ref for arm 1 -- for arm-side ones,
% quarantined included), and the green window lines are that unit's too.
% median_post / sign test / z-score are recomputed in that post window
% ([tons, tons + win_length], baseline as in create_responses_data_parallel);
% the subset's own baseline (mu/sd) and IFR trace are kept. A z-score with
% sd_base = 0 (sparse subset, no baseline spikes) is undefined -> NaN.
%
% OUTPUTS (under out_dir/subset_rasters/<arm>/<subset>/)
%   grapes_blanks.mat (synthetic, the subset only) + the raster page PNGs

p = inputParser;
addParameter(p, 'dir', pwd);
addParameter(p, 'arm', 1);
addParameter(p, 'times', []);
addParameter(p, 'name', 'custom');
addParameter(p, 'side', 'arm');   % custom 'times': whose latency / windows to use, 'gt' (reference) or 'arm'
addParameter(p, 'clip_to_window', false);
addParameter(p, 'out_dir', '');
addParameter(p, 'window', 'defined');   % 'defined' = the run's latency / duration windows, 'default' = its default-window version   % pages go to <out_dir>/<name> instead of <out_dir of the run>/subset_rasters/<arm>/<name>
parse(p, varargin{:});
o = p.Results;

here = fileparts(mfilename('fullpath'));
pipeline_root = fileparts(here);
addpath(fullfile(pipeline_root, 'real_response_analysis'));
addpath(fullfile(pipeline_root, 'responses'));
addpath(genpath(fullfile(pipeline_root, 'codes_emu')));
addpath(genpath(fullfile(pipeline_root, 'useful_functions')));

st = load_validation_state(o.dir);
cfg = st.cfg;
k = arm_index(cfg, o.arm);
arm_name = cfg.arms(k).name;

if ~isempty(o.times)
    L = {struct('name', o.name, 'side', o.side, 'desc', o.name, 'times', o.times(:))};
else
    sf = fullfile(cfg.out_dir, 'subsets', sprintf('subsets_%s.mat', arm_name));
    assert(exist(sf, 'file') == 2, 'plot_subset_rasters: %s not found -- run rerun_validation(''subsets'') first.', sf);
    S = load(sf, 'subsets');
    if ischar(subset_names) && strcmp(subset_names, 'all')
        subset_names = fieldnames(S.subsets);
    elseif ischar(subset_names) && strcmp(subset_names, 'default')
        fn = fieldnames(S.subsets);
        subset_names = [intersect({'gt_only', 'arm_only', 'matched', 'q_not_rescued'}, fn, 'stable'); ...
                        fn(startsWith(fn, 'qnr_'))];
    end
    subset_names = cellstr(subset_names);
    bad = setdiff(subset_names, fieldnames(S.subsets));
    assert(isempty(bad), 'plot_subset_rasters: unknown subset(s) %s; saved: %s', ...
        strjoin(bad, ', '), strjoin(fieldnames(S.subsets)', ', '));
    L = cellfun(@(n) S.subsets.(n), subset_names, 'UniformOutput', false);
end

assert(~isempty(st.ref_tables), ['plot_subset_rasters: no raster tables in %s -- ' ...
    'run rerun_validation(''rasters'') there first.'], st.state_dir);
tables = st.ref_tables;
if strcmp(o.window, 'default')
    assert(isfield(st, 'ref_tables_default') && ~isempty(st.ref_tables_default), ...
        'plot_subset_rasters: no default-window tables in %s (no stimulus has a late latency / long duration)', st.state_dir);
    tables = st.ref_tables_default;
end
old = pwd;
cleanup = onCleanup(@() cd(old));
for i = 1:numel(L)
    s = L{i};
    if strcmp(s.side, 'gt')
        src = tables{1};  src_grapes = fullfile(cfg.gt.dir, cfg.gt.grapes_file);
    else
        src = tables{1 + k};  src_grapes = fullfile(cfg.arms(k).dir, cfg.arms(k).grapes_file);
    end
    sub_dir = fullfile(cfg.out_dir, 'subset_rasters', arm_name, s.name);
    if ~isempty(o.out_dir), sub_dir = fullfile(o.out_dir, s.name); end
    gf = synth_grapes_for_subset(s.times, cfg.ch, cfg.root, src_grapes, sub_dir, s.name, false);
    if isempty(gf)
        fprintf('[subset rasters] %s: no spikes, skipped\n', s.name);
        continue
    end

    % score the subset on its own (its 'mu' raster), every stimulus, then keep stim_list
    cd(sub_dir);
    out = characterize_grapes_responses('channels2plot', cfg.ch, 'stim_list', 'all', ...
        'session_root', cfg.root, 'grapes_files', {'grapes_blanks.mat'}, ...
        'unit_label', {'mu'}, 'save_ifr', true);
    cd(old);
    T = out.grapes_blanks.metrics;
    Tc = T(T.channel_num == cfg.ch & strcmp(cellstr(T.class), 'mu') & ismember(T.stim_number, cfg.stim_list), :);
    [~, pos] = ismember(Tc.stim_number, cfg.stim_list);
    [~, ord] = sort(pos);
    Tc = Tc(ord, :);
    grapes = load(gf);

    % the source unit's latency / duration / post window and raster windows
    [~, ia] = ismember(Tc.stim_number, src.stim_number);
    assert(all(ia > 0), 'plot_subset_rasters: stimuli missing from the reference table');
    for f = {'onset', 'tons', 'dura', 'good_lat'}
        Tc.(f{1}) = src.(f{1})(ia);
    end
    lo = src.win_lo(ia); hi = src.win_hi(ia);
    Tc = rescore_in_ref_window(Tc, grapes.rasters.(sprintf('chan%d', cfg.ch)).mu, ...
        out.grapes_blanks.resp_conf, out.grapes_blanks.rank_config);
    n_in = count_in_windows(s.times, Tc.stim_number, lo, hi, st.onset_map);

    plot_response_window_grapes(grapes, Tc, cfg.ch, 0, n_in, lo, hi, ...
        out.grapes_blanks.rank_config, T.Properties.UserData.ifr_ejex, st.Nscr, ...
        s.name, sub_dir, cfg.root, 'mu', o.clip_to_window, ...
        numel(s.times), [], 'x_jitter', cfg.x_jitter, 'footer', footer_of(src));
    fprintf('[subset rasters] %s (%s): %d spikes, %d in window -> %s\n', s.name, s.desc, numel(s.times), n_in, sub_dir);
end
end

function Tc = rescore_in_ref_window(Tc, unit, rc, rk)
% median_post / sign test / z-score in [tons, tons + win_length] per stimulus,
% baseline (-tmax_median, -tmin_median) -- create_responses_data_parallel's
% counting, with tons taken from the reference unit
    for i = 1:height(Tc)
        tr = unit.stim{Tc.stim_number(i)};
        if iscell(tr), trials = tr(:); else, trials = num2cell(tr, 2); end
        base = cellfun(@(x) sum(x < -rc.tmin_median & x > -rc.tmax_median), trials);
        post = cellfun(@(x) sum(x > Tc.tons(i) & x < Tc.tons(i) + rk.win_length), trials);
        Tc.median_post(i) = median(post);
        Tc.p_value_sign(i) = signtest(base, post, 'Tail', 'left');
    end
    Tc.post_lo = Tc.tons;
    Tc.post_hi = Tc.tons + rk.win_length;
    Tc.base_lo = repmat(-rc.tmax_median, height(Tc), 1);
    Tc.base_hi = repmat(-rc.tmin_median, height(Tc), 1);
    Tc.zscore = (Tc.median_post - Tc.mu_base) ./ (Tc.std_base + 1e-10);
    % sparse subset with no baseline spikes: z is undefined there, not huge
    Tc.zscore(~(Tc.std_base > 0)) = NaN;
    Tc.p_test = Tc.p_value_sign <= rc.psign_thr;
    Tc.min_spk_test = Tc.median_post >= rc.min_spk_median;
end

function f = footer_of(src)
% the full run's window label (run_validation stores it with each raster table)
    f = '';
    if isstruct(src.Properties.UserData) && isfield(src.Properties.UserData, 'footer')
        f = ['full-run windows: ' src.Properties.UserData.footer];
    end
end

function k = arm_index(cfg, arm)
    if isnumeric(arm), k = arm; return; end
    k = find(strcmp({cfg.arms.name}, arm), 1);
    assert(~isempty(k), 'unknown arm %s; arms: %s', arm, strjoin({cfg.arms.name}, ', '));
end
