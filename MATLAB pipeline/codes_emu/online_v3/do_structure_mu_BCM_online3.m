function do_structure_mu_BCM_online3(channels,exp_type,use_blanks, circshiftblanks, is_online, time_pre,time_pos,varargin)
% Arranges the spike trains into matrices for each cluster and stimulus.
% Gets channel/s as input. If no channels are specified tries to read them
% from tile or else goes through all of them.
ipr = inputParser;
addParameter(ipr,'spike_dir','',@ischar)
parse(ipr, varargin{:});
spike_dir = ipr.Results.spike_dir;

begin_time = tic;
fprintf("do_structure_mu_BCM_online3 (use_blanks:%s, circshiftblanks:%s): ", ...
                            mat2str(use_blanks), mat2str(circshiftblanks))
if ~exist('time_pre','var') || isempty(time_pre), time_pre=1e3; end
if ~exist('time_pos','var') || isempty(time_pos), time_pos=2e3; end

load('NSx','NSx');
load stimulus;
load finalevents;
if min(cell2mat({stimulus{1}.ISI}))<0.5
    time_pre=500; time_pos=750;
end

% --- 4-TIER SOURCE RESOLUTION (SPIKES) ---
use_workspace = false;
target_spikes_folder = '';

% Priority 1: String File/Dir Input
if (ischar(spike_dir) || isstring(spike_dir)) && isfolder(spike_dir)
    target_spikes_folder = char(channels);
    fprintf('Priority 1: Using explicit input directory: %s\n', target_spikes_folder);
end

% Priority 2: Workspace Check (Strictly if is_online is true)
if isempty(target_spikes_folder) && is_online
    if evalin('base', 'exist(''index'', ''var'')')
        use_workspace = true;
        fprintf('Priority 2: is_online is true. Using spike index directly from base workspace.\n');
    else
        warning('is_online is true, but ''index'' was not found in the base workspace. Falling back to file search.');
    end
end

% Priority 3: Current Directory (Flat Structure)
if isempty(target_spikes_folder) && ~use_workspace
    if isnumeric(channels) && ~isempty(channels)
        posch = find(arrayfun(@(x) (x.chan_ID==channels(1)), NSx));
        if ~isempty(posch) && exist(fullfile(pwd, sprintf('%s_spikes.mat', NSx(posch(1)).output_name)), 'file')
            target_spikes_folder = pwd;
            fprintf('Priority 3: Found spikes in current working directory.\n');
        end
    end
end

% Priority 4: Timestamped Date Folders
if isempty(target_spikes_folder) && ~use_workspace
    dates_spikes = dir(fullfile(pwd, 'spikes*'));
    dates_spikes = dates_spikes([dates_spikes.isdir]);
    if ~isempty(dates_spikes)
        [~, idx_spk] = max([dates_spikes.datenum]);
        target_spikes_folder = fullfile(pwd, dates_spikes(idx_spk).name);
        fprintf('Priority 4: Locked onto timestamped spikes folder: %s\n', dates_spikes(idx_spk).name);
    else
        error('Could not find spikes in explicit input, workspace, current directory, or timestamped folders.');
    end
end

% Set up Grapes File Name & Path
if use_blanks && circshiftblanks
    grapes_name = 'grapes_blanks_circ.mat';
elseif use_blanks
    grapes_name = 'grapes_blanks.mat';
else
    grapes_name = 'grapes.mat';
end

if use_workspace
    grapes_full_path = fullfile(pwd, grapes_name);
else
    grapes_full_path = fullfile(target_spikes_folder, grapes_name);
end

% Initialize or Load base Grapes
if exist(grapes_full_path, 'file') > 0
    grapes = load(grapes_full_path);
else
    grapes = struct;
    grapes.exp_type = exp_type; 
    grapes.time_pre = time_pre;
    grapes.time_pos = time_pos;
end

load('experiment_properties_online3.mat','experiment','scr_config_cell','scr_end_cell')
spikes = cell(length(channels), 1);
output_names = cell(length(channels), 1);

% Load Data
for i=1:length(channels)
    channel=channels(i);
    posch = find(arrayfun(@(x) (x.chan_ID==channel),NSx));
    output_names{i} = NSx(posch).output_name;    
    
    if use_workspace
        try
            spikes{i} = evalin('base', 'index');
        catch
            warning('Could not load index from workspace for channel %d', channel);
        end
    else
        filename = sprintf('%s_spikes.mat', output_names{i});
        full_file_path = fullfile(target_spikes_folder, filename);
        
        if ~exist(full_file_path,'file') 
            disp([filename ' does not exist in ' target_spikes_folder]);
            channels(i)=[]; 
            continue;
        end
        warning off    
        load(full_file_path,'index');
        spikes{i} = index;
        warning on
    end
end

Nscr = numel(scr_config_cell);
n_scr_ended = numel(scr_end_cell);

if Nscr>n_scr_ended
    warning('some subscreening not ended, using just the completed screenings');
    Nscr = n_scr_ended;
end
ISI_min = Inf;
for scri = 1:Nscr
    ISI_min = min(ISI_min,min(cell2mat({stimulus{scri}.ISI})));
    if use_blanks
        % check if seq_beg_blanks_cell exists, else show warning
        if ~exist('seq_beg_blanks_cell','var') || isempty(seq_beg_blanks_cell) || isempty(seq_beg_blanks_cell{scri})
            error('seq_beg_blanks_cell not found, run extract_blank_on_events_ripple.m from processing_steps to create it.');            
        end
        grapes = update_grapes_blanks(grapes, pics_onset{scri}, ...
                                      seq_beg_blanks_cell{scri}, ...
                                      stimulus{scri}, spikes, channels, ...
                                      output_names, 1, [], ...
                                      scr_config_cell{scri}.pics2use, scri, ...
                                      circshiftblanks, is_online);
    else
        grapes = update_grapes(grapes, pics_onset{scri}, ...
                               stimulus{scri}, spikes, channels, ...
                               output_names, 1, [], ...
                               scr_config_cell{scri}.pics2use, scri, is_online);
    end
   
   fprintf('scr:%d ',scri);
end
fprintf('\n');
grapes.ImageNames = experiment.ImageNames.name;
grapes.ISI_min = ISI_min; 
save(grapes_full_path,'-struct', 'grapes');   
tot_time = toc(begin_time);
fprintf("do_structure_mu_BCM_online3 (use_blanks:%s, circshiftblanks:%s) done in (%0.2f seconds)\n", ...
                                            mat2str(use_blanks), mat2str(circshiftblanks), tot_time)
