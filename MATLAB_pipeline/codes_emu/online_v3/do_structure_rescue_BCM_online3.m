function do_structure_rescue_BCM_online3(channels, use_blanks, circshiftblanks, is_online, varargin)
% DO_STRUCTURE_RESCUE_BCM_ONLINE3
% Build a post-rescue companion grapes file (grapes*_rescue.mat) that has the
% SAME schema / field names as the standard grapes file, so every existing
% consumer works on it unchanged (plot_grapes_as_online,
% create_responses_data_parallel, plot_channel_grapes, revert_grapes,
% compare_grapes_response_trials, characterize_grapes_responses, ...).
%
% Run this AFTER rescue_spikes(channels,'restore',false) and AFTER the normal
% do_structure_sorted_BCM_online3 (which wrote the pre-rescue grapes*.mat).
%
% For every channel it reads times_<name>.mat and writes, into ONE
% self-contained grapes*_rescue.mat (same schema as the standard grapes file):
%   mu                <- cluster_class(:,2)            (post-rescue: all rows)
%   mu_prerescue      <- cluster_class_pre_rescue(:,2) (pre-rescue: all rows)
%   classN            <- cluster_class            (post-rescue: original + rescued)
%   classN_prerescue  <- cluster_class_pre_rescue (the pre-rescue baseline)
%   classN_rescue     <- post spikes not in cluster_class_pre_rescue (rescued only)
% Channels that rescue_spikes left untouched have *_prerescue == the post field.
%
% MU here is the sortable spike population (times-file cluster_class rows,
% class 0 included). In this pipeline the spikes-file `index` that
% do_structure_mu uses is the mask-passed subset, so it equals
% cluster_class_pre_rescue BEFORE rescue - and rescue re-admits quarantined
% spikes into it, so post-rescue MU is genuinely larger. That is why mu is
% rebuilt here (not copied) and mu_prerescue is provided.
%
% Because the pre-rescue baseline travels inside grapes*_rescue.mat, the
% pre/post comparison does NOT depend on grapes.mat still being pre-rescue:
% compare_grapes_response_trials('mode','rescue') reads <unit>_prerescue vs
% <unit> from this one file.
%
% Name-value args (same as do_structure_sorted_BCM_online3):
%   'times_dir'  ''    explicit times folder (else auto-resolved)
%   'spike_dir'  ''    explicit spikes folder (else most-recent spikes*)
%   'time_pre'   1e3
%   'time_pos'   2e3

begin_time = tic;
fprintf("do_structure_rescue_BCM_online3 (use_blanks:%s, circshiftblanks:%s): ", ...
        mat2str(use_blanks), mat2str(circshiftblanks))

ipr = inputParser;
addParameter(ipr, 'times_dir', '', @ischar);
addParameter(ipr, 'spike_dir', '', @ischar);
addParameter(ipr, 'time_pre', 1e3, @isnumeric);
addParameter(ipr, 'time_pos', 2e3, @isnumeric);
parse(ipr, varargin{:});

target_times_folder  = ipr.Results.times_dir;
target_spikes_folder = ipr.Results.spike_dir;
time_pre = ipr.Results.time_pre;
time_pos = ipr.Results.time_pos;

% --- SOURCE RESOLUTION (TIMES) -- mirrors do_structure_sorted_BCM_online3 ---
[~, current_dir_name] = fileparts(pwd);
root = resolve_session_root();
load(fullfile(root, 'NSx.mat'), 'NSx');

if ~isempty(target_times_folder)
    if ~isfolder(target_times_folder), error('Times dir not found: %s', target_times_folder); end
    fprintf('Priority 1: Using user-specified times: %s\n', target_times_folder);
elseif startsWith(current_dir_name, 'times')
    target_times_folder = pwd;
    fprintf('Priority 3: Using current working directory (times subfolder): %s\n', target_times_folder);
elseif isnumeric(channels) && ~isempty(channels) && exist(fullfile(root, ...
        sprintf('times_%s.mat', NSx(find(arrayfun(@(x) (x.chan_ID==channels(1)), NSx),1)).output_name)), 'file')
    target_times_folder = root;
    fprintf('Priority 4: Using session root directory.\n');
else
    dates = dir(fullfile(root, 'times*'));
    dates = dates([dates.isdir]);
    if isempty(dates), error('Could not resolve times directory.'); end
    [~, idx] = max([dates.datenum]);
    target_times_folder = fullfile(root, dates(idx).name);
    fprintf('Priority 5: Using folder: %s\n', dates(idx).name);
end

% --- Resolve spikes folder (only used to locate the base grapes file) ------
if isempty(target_spikes_folder)
    dates_s = dir(fullfile(root, 'spikes*'));
    dates_s = dates_s([dates_s.isdir]);
    if isempty(dates_s), error('No spikes folders found.'); end
    [~, idx_s] = max([dates_s.datenum]);
    associated_spikes_folder = fullfile(root, dates_s(idx_s).name);
else
    associated_spikes_folder = target_spikes_folder;
end

if use_blanks && circshiftblanks
    grapes_name = 'grapes_blanks_circ.mat';
elseif use_blanks
    grapes_name = 'grapes_blanks.mat';
else
    grapes_name = 'grapes.mat';
end

grapes_source_path = fullfile(associated_spikes_folder, grapes_name);
grapes_save_path   = fullfile(target_times_folder, grapes_name);
rescue_save_path   = regexprep(grapes_save_path, '\.mat$', '_rescue.mat');

% --- Load the base grapes (for mu / ImageNames / exp_type / time_pre|pos) ---
if exist(grapes_save_path, 'file') > 0
    grapes = load(grapes_save_path);
elseif exist(grapes_source_path, 'file') > 0
    grapes = load(grapes_source_path);
else
    error(['do_structure_rescue: base grapes file "%s" not found. Run ' ...
           'do_structure_mu_BCM_online3 / do_structure_sorted_BCM_online3 first.'], grapes_name);
end
if ~isfield(grapes, 'time_pre') || isempty(grapes.time_pre), grapes.time_pre = time_pre; end
if ~isfield(grapes, 'time_pos') || isempty(grapes.time_pos), grapes.time_pos = time_pos; end

load(fullfile(root, 'stimulus.mat'));
load(fullfile(root, 'finalevents.mat'));
load(fullfile(root, 'experiment_properties_online3.mat'), 'experiment', 'scr_config_cell', 'scr_end_cell');

if use_blanks && (~exist('seq_beg_blanks_cell', 'var') || isempty(seq_beg_blanks_cell))
    error(['do_structure_rescue: use_blanks requested but seq_beg_blanks_cell is missing. ' ...
           'Run extract_blank_on_events_ripple.m first.']);
end
if ~use_blanks
    seq_beg_blanks_cell = {};
end

% --- Load post / pre / rescued-only spike trains per channel --------------
num_chan     = numel(channels);
spikes_post  = cell(num_chan,1);   classes_post = cell(num_chan,1);
spikes_pre   = cell(num_chan,1);   classes_pre  = cell(num_chan,1);
spikes_resc  = cell(num_chan,1);   classes_resc = cell(num_chan,1);
mu_post      = cell(num_chan,1);   mu_pre       = cell(num_chan,1);
output_names = cell(num_chan,1);
inds_notimes = [];
n_resc_total = 0;
n_ch_rescued = 0;

for i = 1:num_chan
    channel = channels(i);
    posch = find(arrayfun(@(x) (x.chan_ID==channel), NSx));
    output_names{i} = NSx(posch).output_name;

    full_file_path = fullfile(target_times_folder, sprintf('times_%s.mat', output_names{i}));
    if ~exist(full_file_path, 'file')
        disp([full_file_path ' does not exist']);
        inds_notimes = [inds_notimes i];
        continue
    end

    S  = load(full_file_path, 'cluster_class', 'cluster_class_pre_rescue');
    cc = S.cluster_class;
    non0 = cc(:,1) > 0;
    spikes_post{i}  = cc(non0,2).';
    classes_post{i} = cc(non0,1).';
    mu_post{i}      = cc(:,2).';           % MU = all sortable spikes (class 0 incl.)

    has_pre = isfield(S, 'cluster_class_pre_rescue') && ~isempty(S.cluster_class_pre_rescue);
    if has_pre
        ccp = S.cluster_class_pre_rescue;
    else
        ccp = cc;                       % channel never rescued -> pre == post
    end
    non0p = ccp(:,1) > 0;
    spikes_pre{i}  = ccp(non0p,2).';
    classes_pre{i} = ccp(non0p,1).';
    mu_pre{i}      = ccp(:,2).';

    if has_pre
        % Rescued rows are appended verbatim by rescue_spikes, so their spike
        % times match nothing in the pre-rescue table.
        is_new = ~ismember(cc(:,2), ccp(:,2));
    else
        is_new = false(size(cc,1),1);
    end
    sel = is_new & non0;
    spikes_resc{i}  = cc(sel,2).';
    classes_resc{i} = cc(sel,1).';
    n_resc_total = n_resc_total + nnz(sel);
    if nnz(sel) > 0, n_ch_rescued = n_ch_rescued + 1; end
    fprintf('  [%s] non-zero spikes  pre=%d  post=%d  (+%d rescued)\n', ...
            output_names{i}, nnz(non0p), nnz(non0), nnz(sel));
end

channels(inds_notimes)     = [];
spikes_post(inds_notimes)  = [];
classes_post(inds_notimes) = [];
spikes_pre(inds_notimes)   = [];
classes_pre(inds_notimes)  = [];
spikes_resc(inds_notimes)  = [];
classes_resc(inds_notimes) = [];
mu_post(inds_notimes)      = [];
mu_pre(inds_notimes)       = [];
output_names(inds_notimes) = [];

if isempty(channels)
    error('do_structure_rescue: no usable times_*.mat files for the requested channels.');
end
fprintf('%d rescued spike(s) across %d/%d channel(s).\n', ...
        n_resc_total, n_ch_rescued, numel(channels));
if n_resc_total == 0
    warning(['do_structure_rescue: rescue_spikes re-admitted 0 spikes for these ' ...
             'channels; grapes_rescue.mat classN will equal classN_prerescue.']);
end

Nscr = numel(scr_config_cell);
n_scr_ended = numel(scr_end_cell);
if Nscr > n_scr_ended
    warning('some subscreening not ended, using just the completed screenings');
    Nscr = n_scr_ended;
end

upd_args = {use_blanks, circshiftblanks, is_online, channels, output_names, ...
            stimulus, pics_onset, scr_config_cell, seq_beg_blanks_cell};

% Drop any mu/class* rasters carried in from the base file so update_grapes
% (which only ever appends) rebuilds them cleanly from the post-rescue data.
grapes = strip_unit_fields(grapes);
% update_grapes only creates rasters.chanX.mu for a brand-new channel; our
% channels already exist (details survived the strip), so seed an empty one.
for ci = 1:numel(channels)
    chn = ['chan' num2str(channels(ci))];
    if isfield(grapes.rasters, chn)
        grapes.rasters.(chn).mu = struct();
    end
end

% --- mu  <- post-rescue full population -----------------------------------
for scri = 1:Nscr
    grapes = run_update(grapes, 1, mu_post, [], scri, upd_args{:});
    fprintf('mu scr:%d ', scri);
end
fprintf('\n');

% --- classN  <- post-rescue population ------------------------------------
for scri = 1:Nscr
    grapes = run_update(grapes, 0, spikes_post, classes_post, scri, upd_args{:});
    fprintf('post scr:%d ', scri);
end
fprintf('\n');

% --- mu_prerescue / classN_prerescue  <- pre-rescue baseline -----------
grapes = build_suffixed(grapes, '_prerescue', 1, mu_pre,     [],           Nscr, upd_args{:});
grapes = build_suffixed(grapes, '_prerescue', 0, spikes_pre, classes_pre,  Nscr, upd_args{:});

% --- classN_rescue  <- rescued-only spikes -----------------------------
grapes = build_suffixed(grapes, '_rescue',    0, spikes_resc, classes_resc, Nscr, upd_args{:});

grapes.ImageNames = experiment.ImageNames.name;
save(rescue_save_path, '-struct', 'grapes');

tot_time = toc(begin_time);
fprintf("do_structure_rescue_BCM_online3 wrote %s (%0.2f seconds)\n", rescue_save_path, tot_time)
end

% ========================================================================
function g = strip_unit_fields(g)
% Remove mu / classN and their _rescue / _prerescue variants (raster + details
% entries), keeping only details bookkeeping (sr / output_name / ch_label).
    if ~isfield(g, 'rasters'), return; end
    pat = '^(mu|class\d+)(_rescue|_prerescue)?$';
    chs = fieldnames(g.rasters);
    for ci = 1:numel(chs)
        ch = chs{ci};
        fl = fieldnames(g.rasters.(ch));
        hits = fl(~cellfun('isempty', regexp(fl, pat, 'once')));
        for k = 1:numel(hits)
            g.rasters.(ch) = rmfield(g.rasters.(ch), hits{k});
        end
        if isfield(g.rasters.(ch), 'details')
            dfl = fieldnames(g.rasters.(ch).details);
            dhits = dfl(~cellfun('isempty', regexp(dfl, pat, 'once')));
            for k = 1:numel(dhits)
                g.rasters.(ch).details = rmfield(g.rasters.(ch).details, dhits{k});
            end
        end
    end
end

% ========================================================================
function g = build_suffixed(g, suffix, domu, sp, cl, Nscr, varargin)
% Build mu/classN rasters from (sp, cl) in a scratch struct, then graft them
% onto g as <unit><suffix> (+ the matching details entry). update_grapes only
% ever appends, hence the scratch struct + rename rather than writing g direct.
    tmp = struct('time_pre', g.time_pre, 'time_pos', g.time_pos);
    for scri = 1:Nscr
        tmp = run_update(tmp, domu, sp, cl, scri, varargin{:});
        fprintf('%s scr:%d ', suffix, scri);
    end
    fprintf('\n');
    if ~isfield(tmp, 'rasters'), return; end
    chs = fieldnames(tmp.rasters);
    for ci = 1:numel(chs)
        ch  = chs{ci};
        ufl = fieldnames(tmp.rasters.(ch));
        ufl = ufl(~cellfun('isempty', regexp(ufl, '^(mu|class\d+)$', 'once')));
        for k = 1:numel(ufl)
            src = ufl{k};
            dst = [src suffix];
            g.rasters.(ch).(dst) = tmp.rasters.(ch).(src);
            if isfield(tmp.rasters.(ch), 'details') && isfield(tmp.rasters.(ch).details, src)
                g.rasters.(ch).details.(dst) = tmp.rasters.(ch).details.(src);
            else
                g.rasters.(ch).details.(dst) = 0;
            end
        end
    end
end

% ========================================================================
function g = run_update(g, domu, sp, cl, scri, use_blanks, circshiftblanks, ...
                        is_online, channels, output_names, stimulus, pics_onset, ...
                        scr_config_cell, seq_beg_blanks_cell)
    if use_blanks
        g = update_grapes_blanks(g, pics_onset{scri}, seq_beg_blanks_cell{scri}, ...
                                 stimulus{scri}, sp, channels, output_names, domu, ...
                                 cl, scr_config_cell{scri}.pics2use, ...
                                 scri, circshiftblanks, is_online);
    else
        g = update_grapes(g, pics_onset{scri}, ...
                          stimulus{scri}, sp, channels, output_names, domu, ...
                          cl, scr_config_cell{scri}.pics2use, scri, is_online);
    end
end
