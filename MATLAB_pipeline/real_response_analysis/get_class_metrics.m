function m = get_class_metrics(dir_path, grapes_filename, ch, class_num, stim_list, root)
% GET_CLASS_METRICS  Runs characterize_grapes_responses on ONE grapes file
% (in its own folder), scored against the FULL stimulus set (not just
% stim_list) -- mu_base/std_base/IFR_thr are pooled across every stimulus in
% the scoring call (see create_responses_data.m), so restricting to a small
% stim_list would compute a different, noisier baseline/threshold than what
% plot_grapes_as_online.m's raster plots use (which score everything by
% default) -- that mismatch is exactly what showed up as onset/dura not
% matching the raster images. Score 'all', THEN filter the returned rows
% down to stim_list, so onset/dura/ifr_thr match the reference plots exactly
% while still only reporting the stimuli you asked about.
%
% Standalone copy of the local function already used in
% test_rescue_grape_metrics.m -- pulled out (unchanged) so other run scripts
% can call it too.
    % VARIABLES
    %   dir_path/grapes_filename - where the grapes file lives, and its name
    %   ch/class_num/stim_list   - which channel/class/stimuli to report (NOT to score)
    %   root        - session root (has NSx.mat, experiment_properties, ...)
    %   old         - pwd before cd'ing in, restored on exit via onCleanup
    %   out         - characterize_grapes_responses' full return struct
    %   gbase/tag   - grapes filename without extension / the out.(tag) field name
    %   T           - out.(tag).metrics, one row per (channel,class,stim), EVERY stim scored
    %   clname      - 'class<N>' string used to filter T down to this class
    %   sel         - logical row mask: this channel AND this class AND in stim_list
    %   Tc          - T filtered down to just this class's rows, restricted to stim_list
    %   m           - output struct: mean_zscore, mu_base, std_base, n_stims, n_trials, per_stim
    old = pwd;
    cleanup = onCleanup(@() cd(old));
    cd(dir_path);
    out = characterize_grapes_responses('channels2plot', ch, 'stim_list', 'all', ...
                                        'session_root', root, 'grapes_files', {grapes_filename}, ...
                                        'save_ifr', true);   % need the actual IFR trace for the response-window plots
    [~, gbase] = fileparts(grapes_filename);
    tag = matlab.lang.makeValidName(gbase);
    T = out.(tag).metrics;
    m.rank_config = out.(tag).rank_config;
    m.ejex        = T.Properties.UserData.ifr_ejex;
    clname = sprintf('class%d', class_num);
    sel_class = T.channel_num == ch & strcmp(cellstr(T.class), clname);
    Tc = T(sel_class & ismember(T.stim_number, stim_list), :);
    % Rows in the caller's stim_list order (the metrics table itself is in
    % numerical stim order) -- this is the order the raster pages plot in.
    [~, pos] = ismember(Tc.stim_number, stim_list);
    [~, ord] = sort(pos);
    Tc = Tc(ord, :);
    m.mean_zscore = mean(Tc.zscore, 'omitnan');
    m.mu_base     = Tc.mu_base(1);
    m.std_base    = Tc.std_base(1);
    m.n_stims     = height(Tc);
    m.n_trials    = sum(Tc.ntrials);   % total trials scored, summed across stim_list -- for rate calc
    m.per_stim    = Tc;                % per-(class,stim) rows, stim_list only -- for the dynamic window
    m.per_stim_all = T(sel_class, :);  % SAME class, but EVERY stimulus scored this call -- reuse this for the
                                        % overall/session-wide summary later instead of re-running
                                        % characterize_grapes_responses; it's already sitting here for free.
end
