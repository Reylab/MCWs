function do_structure_sorted_BCM_online3(channels, use_blanks, circshiftblanks, is_online,varargin)
% function do_structure_sorted(channels)
% Arranges the spike trains into matrices for each cluster and stimulus.
% Gets channel/s as input. If no channels are specified goes through all of
% them.
% MJI: 6/7/2010
begin_time = tic;
ipr = inputParser;
addParameter(ipr, 'times_dir', '', @ischar);
addParameter(ipr, 'spike_dir', '', @ischar);
addParameter(ipr, 'time_pre', 1e3, @isnumeric);
addParameter(ipr, 'time_pos', 2e3, @isnumeric);
parse(ipr, varargin{:});

target_times_folder = ipr.Results.times_dir;
target_spikes_folder = ipr.Results.spike_dir; % Use this later
time_pre = ipr.Results.time_pre;
time_pos = ipr.Results.time_pos;

% --- 5-TIER SOURCE RESOLUTION (TIMES) ---
use_workspace = false;
[~, current_dir_name] = fileparts(pwd);

root = resolve_session_root();
load(fullfile(root, 'NSx.mat'), 'NSx');

% Priority 1: User-Provided Input
if ~isempty(target_times_folder)
    if ~isfolder(target_times_folder), error('Times dir not found: %s', target_times_folder); end
    fprintf('Priority 1: Using user-specified times: %s\n', target_times_folder);

% Priority 2: Workspace
elseif is_online && evalin('base', 'exist(''cluster_class'', ''var'')')
    use_workspace = true;
    fprintf('Priority 2: Using base workspace.\n');

% Priority 3: Current Directory (Subfolder Check)
elseif startsWith(current_dir_name, 'times')
    target_times_folder = pwd;
    fprintf('Priority 3: Using current working directory (times subfolder): %s\n', target_times_folder);

% Priority 4: Current Directory (Flat Structure)
elseif isnumeric(channels) && ~isempty(channels) && exist(fullfile(root, ...
    sprintf('times_%s.mat', NSx(find(arrayfun(@(x) (x.chan_ID==channels(1)), NSx),1)).output_name)), 'file')
    target_times_folder = root;
    fprintf('Priority 4: Using session root directory.\n');

% Priority 5: Timestamped Folder
else
    dates = dir(fullfile(root, 'times*'));
    dates = dates([dates.isdir]);
    if isempty(dates), error('Could not resolve times directory.'); end
    [~, idx] = max([dates.datenum]);
    target_times_folder = fullfile(root, dates(idx).name);
    fprintf('Priority 5: Using folder: %s\n', dates(idx).name);
end

% --- Resolve Spikes Folder (Use Input or Default to Most Recent) ---
if isempty(target_spikes_folder)
    dates_s = dir(fullfile(root, 'spikes*'));
    dates_s = dates_s([dates_s.isdir]);
    if isempty(dates_s), error('No spikes folders found.'); end
    [~, idx_s] = max([dates_s.datenum]);
    associated_spikes_folder = fullfile(root, dates_s(idx_s).name);
else
    associated_spikes_folder = target_spikes_folder;
end
fprintf('Using spikes folder: %s\n', associated_spikes_folder);


if use_blanks && circshiftblanks
    grapes_name = 'grapes_blanks_circ.mat';
elseif use_blanks
    grapes_name = 'grapes_blanks.mat';
else
    grapes_name = 'grapes.mat';
end

grapes_source_path = fullfile(associated_spikes_folder, grapes_name);
if use_workspace
    grapes_save_path = fullfile(pwd, grapes_name);
else
    grapes_save_path = fullfile(target_times_folder, grapes_name);
end

% Load base grapes
if exist(grapes_source_path, 'file') > 0
    grapes = load(grapes_source_path);
elseif exist(grapes_save_path, 'file') > 0
    grapes = load(grapes_save_path);
else
    warning('Base grapes file not found. Initializing empty structure.');
    grapes = struct;
    grapes.time_pre = time_pre;
    grapes.time_pos = time_pos;
end

load(fullfile(root, 'stimulus.mat'));
load(fullfile(root, 'finalevents.mat'));
load(fullfile(root, 'experiment_properties_online3.mat'), 'experiment', 'scr_config_cell', 'scr_end_cell');

num_chan = numel(channels);
spikes = cell(num_chan,1);
classes = cell(num_chan,1);
output_names = cell(num_chan,1);
inds_notimes = [];

% Load Data
for i=1:num_chan
    channel=channels(i);
    posch = find(arrayfun(@(x) (x.chan_ID==channel),NSx));
    output_names{i} = NSx(posch).output_name;
    
    if use_workspace
        try
            cluster_class = evalin('base', 'cluster_class');
            non0 = cluster_class(:,1)>0;
            spikes{i} = cluster_class(non0,2)'; 
            classes{i} = cluster_class(non0,1)'; 
        catch
            warning('Could not load cluster_class from workspace for channel %d', channel);
            inds_notimes = [inds_notimes i];
        end
    else
        filename = sprintf('times_%s.mat', output_names{i});
        full_file_path = fullfile(target_times_folder, filename);
        
        if ~exist(full_file_path,'file') 
            disp([filename ' does not exist in ' target_times_folder]);
            inds_notimes=[inds_notimes i]; 
            continue
        end
        load(full_file_path,'cluster_class');
        non0 = cluster_class(:,1)>0;
        spikes{i} = cluster_class(non0,2)'; 
        classes{i} = cluster_class(non0,1)'; 
    end
end

channels(inds_notimes) = [];
spikes(inds_notimes) = [];
classes(inds_notimes) = [];
output_names(inds_notimes) = [];


Nscr = numel(scr_config_cell);

n_scr_ended = numel(scr_end_cell);
if Nscr>n_scr_ended
    warning('some subscreening not ended, using just the completed screenings');
    Nscr = n_scr_ended;
end

for scri = 1:Nscr
    if use_blanks
        grapes = update_grapes_blanks(grapes,pics_onset{scri}, seq_beg_blanks_cell{scri}, ...
                                      stimulus{scri}, spikes, channels, output_names, 0, ...
                                      classes, scr_config_cell{scri}.pics2use, ...
                                      scri, circshiftblanks, is_online);
    else
        grapes = update_grapes(grapes,pics_onset{scri}, ...
                               stimulus{scri}, spikes, channels, output_names, 0, ...
                               classes, scr_config_cell{scri}.pics2use, scri, is_online);
    end

   fprintf('scr:%d ',scri);
end
fprintf('\n');
% grapes = update_grapes_with_blank_on_spikes(blank_on_onset, grapes, spikes, classes, channels);
grapes.ImageNames = experiment.ImageNames.name;
save(grapes_save_path,'-struct', 'grapes');    
% save(grapes_name,"-v7.3",'-struct', 'grapes');
tot_time = toc(begin_time);
fprintf("do_structure_sorted_BCM_online3 (use_blanks:%s, circshiftblanks:%s) done in (%0.2f seconds)\n", ...
                                                mat2str(use_blanks), mat2str(circshiftblanks), tot_time)
