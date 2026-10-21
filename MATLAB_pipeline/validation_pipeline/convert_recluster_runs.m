function T = convert_recluster_runs(runs_root, sc_root, varargin)
% CONVERT_RECLUSTER_RUNS  Match Python runs (one folder per channel, e.g.
% p11_EMU-017_ch328) to their single-channel session folders and convert
% each run's two top-level times files into MATLAB arm folders there
% (build_python_arm_from_times: adapted times/spikes + grapes).
%
%   T = convert_recluster_runs('/mnt/data0/recluster_runs', ...
%         '/mnt/data0/MCWs Sessions/matlab validation/single channel')
%
% Per run folder, the top-level times_<date>_<time>/times_<channel>.mat files:
%   no quarantine.rescued  -> clustering only      -> arm '<prefix>_clustered'
%   quarantine.rescued     -> + artifact rescue    -> arm '<prefix>_rescued'
% second_pass/ is NOT converted here (pass 'second_pass', true to add it as
% '<prefix>_second_pass', or a subfolder pattern under the run folder, e.g.
% '10_12_reclust_test/second_pass_*/*', to convert that second pass instead;
% its arm is named 'python_recluster_<second-pass date>_second_pass').
%
% Matching: pNN_EMU-EEE_chCCC -> the one single-channel folder whose name
% contains 'pt NN emu EEE ch CCC'. Checked before converting: the channel's
% NC5 name equals the times file's, and the session's stimulus onsets
% (finalevents.pics_onset) and experiment_properties_online3.mat equal the
% source recording's (from the run's batch.log), when the source has them.
% exp_type comes from the folder's hand-selected grapes.
%
% Name-value:
%   'prefix'       arm name prefix, default 'python_recluster_<run date>'
%   'second_pass'  also convert second_pass/times_* (true), or the second
%                  pass in this subfolder pattern (char), default false
%   'top_level'    convert the two top-level times files, default true
%   'runs'         subset of run folder names, default all
%   'dry_run'      match and check only, default false
%   'arm_subdir'   put the arms in this subfolder of the channel folder, default '' (the channel folder),
%                  e.g. 'python second pass 10_19'
%
% OUTPUT
%   T - one row per (run, arm): run, folder, channel, arm, source, status
%   also written to <sc_root>/recluster_conversion_log.csv

p = inputParser;
addParameter(p, 'prefix', '');
addParameter(p, 'second_pass', false);
addParameter(p, 'top_level', true);
addParameter(p, 'runs', {});
addParameter(p, 'dry_run', false);
addParameter(p, 'arm_subdir', '');
parse(p, varargin{:});
o = p.Results;

here = fileparts(mfilename('fullpath'));
pipeline_root = fileparts(here);
addpath(fullfile(pipeline_root, 'real_response_analysis'));
addpath(genpath(fullfile(pipeline_root, 'codes_emu')));
addpath(genpath(fullfile(pipeline_root, 'useful_functions')));
old = pwd; cleanup = onCleanup(@() cd(old));

runs = dir(runs_root);
runs = runs([runs.isdir] & ~startsWith({runs.name}, '.'));
if ~isempty(o.runs), runs = runs(ismember({runs.name}, cellstr(o.runs))); end
sc = dir(sc_root); sc = sc([sc.isdir] & ~startsWith({sc.name}, '.'));

rows = {};
for r = runs'
    tok = regexp(r.name, '^p(\d+)_EMU-(\d+)_ch(\d+)', 'tokens', 'once');
    if isempty(tok), continue; end
    [pt, emu, ch] = deal(str2double(tok{1}), str2double(tok{2}), str2double(tok{3}));
    key = sprintf('pt %d emu %d ch %d', pt, emu, ch);
    hit = sc(~cellfun(@isempty, regexp({sc.name}, ['(^|\s)' key '(\s|$)'], 'once')));
    run_dir = fullfile(runs_root, r.name);
    if numel(hit) ~= 1
        rows(end+1, :) = {r.name, '', ch, '', '', sprintf('%d folders match "%s"', numel(hit), key)}; %#ok<AGROW>
        continue
    end
    root = fullfile(sc_root, hit.name);

    % same recording / channel?
    nc5 = dir(fullfile(root, '*.NC5'));
    tf = [dir(fullfile(run_dir, 'times_*', 'times_*.mat'))];
    msg = check_session(run_dir, root, nc5, tf);
    if ~isempty(msg)
        rows(end+1, :) = {r.name, hit.name, ch, '', '', ['NOT converted: ' msg]}; %#ok<AGROW>
        continue
    end
    g = dir(fullfile(root, 'hand selected', 'grapes*.mat'));
    exp_type = getfield(load(fullfile(g(1).folder, g(1).name), 'exp_type'), 'exp_type'); %#ok<GFLD>

    prefix = o.prefix;
    if isempty(prefix)
        d = regexp(tf(1).folder, 'times_(\d{4})-(\d{2})-(\d{2})', 'tokens', 'once');
        prefix = sprintf('python_recluster_%s%s%s', d{:});
    end
    jobs = cell(0, 2);
    todo = tf;
    if ~o.top_level, todo = tf([]); end
    for f = todo'
        Q = load(fullfile(f.folder, f.name), 'quarantine');
        if isfield(Q, 'quarantine') && isfield(Q.quarantine, 'rescued')
            jobs(end+1, :) = {[prefix '_rescued'], fullfile(f.folder, f.name)}; %#ok<AGROW>
        else
            jobs(end+1, :) = {[prefix '_clustered'], fullfile(f.folder, f.name)}; %#ok<AGROW>
        end
    end
    if ischar(o.second_pass) || isstring(o.second_pass)
        sp = dir(fullfile(run_dir, char(o.second_pass), 'times_*', 'times_*.mat'));
        for f = sp'
            d = regexp(f.folder, 'times_(\d{4})-(\d{2})-(\d{2})', 'tokens', 'once');
            jobs(end+1, :) = {sprintf('python_recluster_%s%s%s_second_pass', d{:}), fullfile(f.folder, f.name)}; %#ok<AGROW>
        end
    elseif o.second_pass
        sp = dir(fullfile(run_dir, 'second_pass', 'times_*', 'times_*.mat'));
        for f = sp', jobs(end+1, :) = {[prefix '_second_pass'], fullfile(f.folder, f.name)}; end %#ok<AGROW>
    end
    if numel(unique(jobs(:,1))) ~= size(jobs, 1)
        rows(end+1, :) = {r.name, hit.name, ch, '', '', 'NOT converted: two times files map to the same arm'}; %#ok<AGROW>
        continue
    end

    for j = 1:size(jobs, 1)
        status = 'matched (dry run)';
        if ~o.dry_run
            try
                build_python_arm_from_times('root', root, 'channel', ch, 'exp_type', exp_type, ...
                    'arm_name', fullfile(o.arm_subdir, jobs{j,1}), 'python_times_file', jobs{j,2});
                status = 'converted';
            catch ME
                status = ['FAILED: ' ME.message];
            end
            cd(old);
        end
        rows(end+1, :) = {r.name, hit.name, ch, fullfile(o.arm_subdir, jobs{j,1}), jobs{j,2}, status}; %#ok<AGROW>
        fprintf('[convert] %s -> %s / %s: %s\n', r.name, hit.name, jobs{j,1}, status);
    end
end
T = cell2table(rows, 'VariableNames', {'run', 'folder', 'channel', 'arm', 'source', 'status'});
if ~o.dry_run
    % update the log: replace this call's (run, arm) rows, keep the others
    lf = fullfile(sc_root, 'recluster_conversion_log.csv');
    if exist(lf, 'file')
        old_T = readtable(lf, 'Delimiter', ',', 'TextType', 'char');
        keep = ~ismember(strcat(old_T.run, '|', old_T.arm), strcat(T.run, '|', T.arm));
        T_all = [old_T(keep, :); T];
    else
        T_all = T;
    end
    writetable(T_all, lf);
end
end

function msg = check_session(run_dir, root, nc5, tf)
    msg = '';
    if numel(nc5) ~= 1, msg = sprintf('%d NC5 files in folder', numel(nc5)); return; end
    [~, nm] = fileparts(nc5.name);
    if numel(tf) ~= 2, msg = sprintf('%d top-level times files (expected 2)', numel(tf)); return; end
    if ~all(strcmp({tf.name}, ['times_' nm '.mat']))
        msg = sprintf('times file %s does not match channel %s', tf(1).name, nm); return;
    end
    src = regexp(fileread(fullfile(run_dir, 'batch.log')), "argv: \['[^']*', '([^']*)'", 'tokens', 'once');
    if isempty(src), return; end
    src = src{1};
    a = load(fullfile(root, 'finalevents.mat'), 'pics_onset');
    if exist(fullfile(src, 'finalevents.mat'), 'file')
        b = load(fullfile(src, 'finalevents.mat'), 'pics_onset');
        if ~isequaln(a.pics_onset, b.pics_onset), msg = ['stimulus onsets differ from ' src]; return; end
    end
    if exist(fullfile(src, 'experiment_properties_online3.mat'), 'file')
        x = dir(fullfile(root, 'experiment_properties_online3.mat'));
        y = dir(fullfile(src, 'experiment_properties_online3.mat'));
        if x.bytes ~= y.bytes, msg = ['experiment properties differ from ' src]; return; end
    end
end
