function do_structure_sorted_BCM_online3(channels, use_blanks, circshiftblanks, is_online,varargin)
% function do_structure_sorted(channels)
% Arranges the spike trains into matrices for each cluster and stimulus.
% Gets channel/s as input. If no channels are specified goes through all of
% them.
% MJI: 6/7/2010

ipr = inputParser;
addParameter(ipr,'spike_dir','',@ischar)
parse(ipr, varargin{:});

spike_dir = ipr.Results.spike_dir;

begin_time = tic;
fprintf("do_structure_sorted_BCM_online3 (use_blanks:%s, circshiftblanks:%s): ", ...
                            mat2str(use_blanks), mat2str(circshiftblanks))
if ~exist('time_pre','var') || isempty(time_pre), time_pre=1e3; end
if ~exist('time_pos','var') || isempty(time_pos), time_pos=2e3; end

load('NSx','NSx');

% --- 4-TIER SOURCE RESOLUTION (TIMES & SPIKES ALIGNMENT) ---
use_workspace = false;
target_times_folder = '';

% Priority 1: String File/Dir Input
if (ischar(spike_dir) || isstring(spike_dir)) && isfolder(spike_dir)
    target_times_folder = spike_dir;
    fprintf('Priority 1: Using explicit input directory: %s\n', target_times_folder);
end

% Priority 2: Workspace Check (Strictly if is_online is true)
if isempty(target_times_folder) && is_online
    if evalin('base', 'exist(''cluster_class'', ''var'')')
        use_workspace = true;
        fprintf('Priority 2: is_online is true. Using clustered data directly from base workspace.\n');
    else
        warning('is_online is true, but ''cluster_class'' was not found in the base workspace. Falling back to file search.');
    end
end

% Priority 3: Current Directory (Flat Structure)
if isempty(target_times_folder) && ~use_workspace
    if isnumeric(channels) && ~isempty(channels)
        posch = find(arrayfun(@(x) (x.chan_ID==channels(1)), NSx));
        if ~isempty(posch) && exist(fullfile(pwd, sprintf('times_%s.mat', NSx(posch(1)).output_name)), 'file')
            target_times_folder = pwd;
            fprintf('Priority 3: Found times in current working directory.\n');
        end
    end
end

% Priority 4: Timestamped Date Folders
if isempty(target_times_folder) && ~use_workspace
    dates_times = dir(fullfile(pwd, 'times*'));
    dates_times = dates_times([dates_times.isdir]);
    if ~isempty(dates_times)
        [~, idx_times] = max([dates_times.datenum]);
        target_times_folder_name = dates_times(idx_times).name;
        target_times_folder = fullfile(pwd, target_times_folder_name);
        fprintf('Priority 4: Locked onto timestamped times folder: %s\n', target_times_folder_name);
    else
        error('Could not find times in explicit input, workspace, current directory, or timestamped folders.');
    end
end

% Resolve Associated Spikes Folder for Grapes loading
if use_workspace || strcmp(target_times_folder, pwd)
    associated_spikes_folder = pwd;
else
    [~, times_dir_name] = fileparts(target_times_folder);
    timestamp_suffix = strrep(times_dir_name, 'times', ''); 
    associated_spikes_folder = fullfile(pwd, ['spikes' timestamp_suffix]);
end

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

load stimulus;
load finalevents;
load('experiment_properties_online3.mat','experiment','scr_config_cell','scr_end_cell')

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
