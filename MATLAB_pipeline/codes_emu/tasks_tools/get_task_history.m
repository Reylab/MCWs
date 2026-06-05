function [emu_num, run_num, csv_path] = get_task_history(patient_id, task_name, rec_metadata_path, varargin)
% GET_TASK_HISTORY Manage task history CSV and return EMU/run numbers
%
% This function manages a CSV file tracking all task runs for a patient.
% It handles creation, scanning existing folders, and determining the next
% EMU and run numbers for a given task.
%
% Usage:
%   [emu_num, run_num] = get_task_history(patient_id, task_name, rec_metadata_path)
%   [emu_num, run_num] = get_task_history(patient_id, task_name, rec_metadata_path, 'scan_paths', {paths})
%   [emu_num, run_num, csv_path] = get_task_history(patient_id, task_name, rec_metadata_path, 'register', true)
%   [emu_num, run_num] = get_task_history(..., 'rescan', true)  % Rescan folders to update CSV
%
% Inputs:
%   patient_id        - Subject ID string (e.g., 'MCW-FH_001')
%   task_name         - Task name string (e.g., 'DynamicScr', 'OnlineMiniScr')
%   rec_metadata_path - Path to rec_metadata folder where CSV is stored
%
% Optional Parameters:
%   'scan_paths'      - Cell array of paths to scan for existing EMU folders
%                       (scans on creation or rescan)
%   'register'        - If true, register a new run as 'started' (default: false)
%   'status'          - Status to set: 'started', 'completed', 'aborted' (default: 'started')
%   'update_status'   - If true, update status of current emu/run (default: false)
%   'n_trials'        - Number of trials completed (for status update)
%   'experiment_folder' - Experiment folder path (for status update)
%   'notes'           - Notes string (for status update)
%   'force_emu'       - Force specific EMU number (optional)
%   'force_run'       - Force specific run number (optional)
%   'rescan'          - If true, rescan folders and merge with existing CSV (default: false)
%
% Outputs:
%   emu_num  - EMU number for this session
%   run_num  - Run number for this task within the EMU session
%   csv_path - Full path to the Task History CSV file
%
% CSV Columns:
%   patient_id, emu_num, task, run_num, status, start_time, end_time,
%   duration_min, n_trials, experiment_folder, notes
%
% Author: ReyLab

    % Parse inputs
    p = inputParser;
    addRequired(p, 'patient_id', @ischar);
    addRequired(p, 'task_name', @ischar);
    addRequired(p, 'rec_metadata_path', @ischar);
    addParameter(p, 'scan_paths', {}, @iscell);
    addParameter(p, 'register', false, @islogical);
    addParameter(p, 'status', 'started', @ischar);
    addParameter(p, 'update_status', false, @islogical);
    addParameter(p, 'n_trials', [], @isnumeric);
    addParameter(p, 'experiment_folder', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'notes', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'force_emu', [], @isnumeric);
    addParameter(p, 'force_run', [], @isnumeric);
    addParameter(p, 'rescan', false, @islogical);  % Rescan folders even if CSV exists
    addParameter(p, 'acq_folder', '', @ischar);  % Acquisition folder to check for recording files
    addParameter(p, 'backup_path', '', @ischar);  % Backup path for auto-moving files
    parse(p, patient_id, task_name, rec_metadata_path, varargin{:});
    opts = p.Results;
    
    % Define CSV path
    csv_filename = sprintf('%s_Task_History.csv', patient_id);
    csv_path = fullfile(rec_metadata_path, csv_filename);
    
    % CSV column headers
    headers = {'patient_id', 'emu_num', 'task', 'run_num', 'status', ...
               'start_time', 'end_time', 'duration_min', 'n_trials', ...
               'experiment_folder', 'notes'};
    
    % Check if CSV exists, if not create it (or rescan if requested)
    if ~isfile(csv_path) || opts.rescan
        if opts.rescan && isfile(csv_path)
            fprintf('Rescanning folders to update Task History CSV: %s\n', csv_filename);
        else
            fprintf('Creating new Task History CSV: %s\n', csv_filename);
        end
        create_csv_from_folders(csv_path, headers, patient_id, opts.scan_paths);
    end
    
    % Read current CSV data
    try
        data = readtable(csv_path, 'TextType', 'string', 'Delimiter', ',');
    catch
        % Empty or corrupted file - recreate
        create_csv_from_folders(csv_path, headers, patient_id, opts.scan_paths);
        data = readtable(csv_path, 'TextType', 'string', 'Delimiter', ',');
    end
    
    % Ensure all columns exist
    for i = 1:length(headers)
        if ~ismember(headers{i}, data.Properties.VariableNames)
            if ismember(headers{i}, {'emu_num', 'run_num', 'duration_min', 'n_trials'})
                data.(headers{i}) = nan(height(data), 1);
            else
                data.(headers{i}) = repmat("", height(data), 1);
            end
        end
    end
    
    % Validate existing entries - mark invalid "started" entries that have no folder/files
    % This prevents counting runs that never actually started
    % Skip validation when update_status is true (force-completing a task)
    if height(data) > 0 && ~opts.update_status
        needs_save = false;
        for row_idx = 1:height(data)
            % Handle missing status values
            status_val = data.status(row_idx);
            if ismissing(status_val)
                continue
            end
            
            % Only check entries with status 'started' or 'invalid_empty_folder' (not completed/error/aborted)
            if any(strcmpi(status_val, {'started', 'invalid_empty_folder','invalid_no_files'}))
                has_files = false;
                
                % Get row info for this entry
                emu_num_row = data.emu_num(row_idx);
                task_row = char(data.task(row_idx));
                run_num_row = data.run_num(row_idx);
                
                % First check: Look for recording files in acquisition folder
                % Files are named like EMU-XXX_subj-YYY_task-ZZZ_run-WW_RIP.ns5
                if ~isempty(opts.acq_folder) && isfolder(opts.acq_folder)

                    % Build file pattern to search for recording files
                    % For gaps task, only check by EMU (internal run nums are separate)
                    if strcmpi(task_row, 'gaps')
                        file_pattern = sprintf('EMU-%.3d_subj-%s_task-*%s*', ...
                                              emu_num_row, patient_id, task_row);
                    else
                        file_pattern = sprintf('EMU-%.3d_subj-%s_task-*%s*_run-%.2d*', ...
                                              emu_num_row, patient_id, task_row, run_num_row);
                    end

                    acq_files = dir(fullfile(opts.acq_folder, file_pattern));
                    
                    if ~isempty(acq_files)
                        has_files = true;
                        if strcmpi(task_row, 'gaps')
                            fprintf('Found %d recording files in acq folder for EMU-%d %s\n', ...
                                    length(acq_files), emu_num_row, task_row);
                        else
                            fprintf('Found %d recording files in acq folder for EMU-%d %s run-%d\n', ...
                                    length(acq_files), emu_num_row, task_row, run_num_row);
                        end
                    end
                end
                
                % Second check: Look in experiment folder if provided
                if ~has_files && ismember('experiment_folder', data.Properties.VariableNames)
                    folder_val = data.experiment_folder(row_idx);
                    if ~ismissing(folder_val)
                        folder_path = char(folder_val);
                        if ~isempty(folder_path) && strlength(folder_path) > 0 && isfolder(folder_path)
                            folder_contents = dir(folder_path);
                            folder_contents = folder_contents(~ismember({folder_contents.name}, {'.', '..'}));
                            if ~isempty(folder_contents)
                                has_files = true;
                            end
                        end
                    end
                end
                
                % Mark entry based on whether files were found
                if has_files
                    % Files exist - task ran but status wasn't updated
                    % Try to move files to backup folder if backup_path is provided
                    moved_files = false;
                    backup_dest = '';
                    
                    if ~isempty(opts.backup_path) && ~isempty(opts.acq_folder)
                        try
                            % Create backup folder structure
                            emu_backup_folder = fullfile(opts.backup_path, patient_id, 'EMU');
                            if ~isfolder(emu_backup_folder)
                                mkdir(emu_backup_folder);
                            end
                            
                            % Create task folder name
                            if strcmpi(task_row, 'gaps')
                                task_folder_name = sprintf('EMU-%.3d_subj-%s_task-%s', emu_num_row, patient_id, task_row);
                            else
                                task_folder_name = sprintf('EMU-%.3d_subj-%s_task-RSVP%s_run-%.2d', emu_num_row, patient_id, task_row, run_num_row);
                            end
                            backup_dest = fullfile(emu_backup_folder, task_folder_name);
                            
                            if ~isfolder(backup_dest)
                                mkdir(backup_dest);
                            end
                            
                            % Move files from acq folder to backup
                            for f_idx = 1:length(acq_files)
                                src_file = fullfile(opts.acq_folder, acq_files(f_idx).name);
                                if isfile(src_file)
                                    movefile(src_file, backup_dest);
                                end
                            end
                            
                            moved_files = true;
                            fprintf('Auto-moved %d files to: %s\n', length(acq_files), backup_dest);
                            
                            % Update experiment_folder in CSV
                            if isnumeric(data.experiment_folder)
                                data.experiment_folder = strings(height(data), 1);
                            end
                            data.experiment_folder(row_idx) = string(backup_dest);
                        catch ME_move
                            fprintf('Warning: Failed to auto-move files: %s\n', ME_move.message);
                        end
                    end
                    
                    % Mark as recovered
                    if moved_files
                        data.status(row_idx) = "completed_recovered";
                        data.notes(row_idx) = sprintf("Auto-recovered and moved to %s", backup_dest);
                    else
                        data.status(row_idx) = "Recording File exist, Folder was not Created";
                        data.notes(row_idx) = "Recording files found but status was started - auto-recovered";
                    end
                    needs_save = true;
                    
                    if strcmpi(task_row, 'gaps')
                        fprintf('Recovering entry: EMU-%d %s (files found, marking completed)\n', ...
                                data.emu_num(row_idx), data.task(row_idx));
                    else
                        fprintf('Recovering entry: EMU-%d %s run-%d (files found, marking completed)\n', ...
                                data.emu_num(row_idx), data.task(row_idx), data.run_num(row_idx));
                    end
                else
                    % No files found - mark as invalid
                    data.status(row_idx) = "invalid_no_files";
                    data.notes(row_idx) = "No recording files found - entry ignored";
                    needs_save = true;
                    if strcmpi(task_row, 'gaps')
                        fprintf('Marking invalid entry: EMU-%d %s (no files found)\n', ...
                                data.emu_num(row_idx), data.task(row_idx));
                    else
                        fprintf('Marking invalid entry: EMU-%d %s run-%d (no files found)\n', ...
                                data.emu_num(row_idx), data.task(row_idx), data.run_num(row_idx));
                    end
                end
            end
        end
        if needs_save
            writetable(data, csv_path);
        end
    end
    
    % Filter to only valid entries for determining EMU/run numbers
    % Exclude entries with status starting with "invalid"
    if height(data) > 0
        valid_mask = ~startsWith(data.status, "invalid", 'IgnoreCase', true);
        valid_data = data(valid_mask, :);
    else
        valid_data = data;
    end
    
    % Determine EMU and run numbers (using only valid entries)
    if ~isempty(opts.force_emu)
        emu_num = opts.force_emu;
    else
        % Get highest EMU number from valid history and INCREMENT
        % (each new session/task gets a new EMU, matching original behavior)
        if height(valid_data) > 0 && any(~isnan(valid_data.emu_num))
            emu_num = max(valid_data.emu_num) + 1;
        else
            emu_num = 1;
        end
    end
    
    if ~isempty(opts.force_run)
        run_num = opts.force_run;
    else
        % Get highest run number for this task across ALL EMUs (not just current EMU)
        % This ensures run numbers continue to increment across sessions
        if height(valid_data) > 0
            task_mask = strcmpi(valid_data.task, task_name);
            
            if any(task_mask)
                run_num = max(valid_data.run_num(task_mask)) + 1;
            else
                run_num = 1;
            end
        else
            run_num = 1;
        end
    end
    
    % Handle status update for existing entry
    if opts.update_status
        task_mask = strcmpi(data.task, task_name);
        emu_mask = data.emu_num == emu_num;
        run_mask = data.run_num == run_num;
        row_mask = task_mask & emu_mask & run_mask;
        
        if any(row_mask)
            row_idx = find(row_mask, 1, 'last');
            data.status(row_idx) = string(opts.status);
            
            % Set end time - handle various column types
            end_dt = datetime('now');
            end_time_str = string(datetime('now', 'Format', 'yyyy-MM-dd''T''HH:mm:ss.SSS'));
            
            % Handle end_time column based on its type
            if isdatetime(data.end_time)
                % Column is datetime - assign datetime value
                data.end_time(row_idx) = end_dt;
            elseif isnumeric(data.end_time)
                % Column is numeric (NaN) - convert to string column
                data.end_time = strings(height(data), 1);
                data.end_time(row_idx) = end_time_str;
            else
                % Column is string/cell - assign string value
                data.end_time(row_idx) = end_time_str;
            end
            
            % Convert duration_min column to double if needed
            if ~isnumeric(data.duration_min)
                data.duration_min = nan(height(data), 1);
            end
           
            
            % Calculate duration - safely handle various value types
            start_val = data.start_time(row_idx);
 
            try
                % Handle numeric (NaN) start_time
                if isnumeric(start_val) && isnan(start_val)
                    fprintf('Warning: start_time is NaN, cannot calculate duration\n');
                elseif isdatetime(start_val)
                    % start_val is already a datetime - use it directly
                    if ~isnat(start_val)
                        duration_val = round(minutes(end_dt - start_val), 1);
                        data.duration_min(row_idx) = duration_val;
                   
                    else
                        fprintf('Warning: start_time is NaT (not-a-time)\n');
                    end
                elseif isstring(start_val) || ischar(start_val)
                    start_str = char(start_val);
              
                    if ~isempty(start_str) && ~strcmp(start_str, '') && ~strcmp(start_str, 'NaN')
                        % Try multiple datetime formats
                        try
                            start_dt = datetime(start_str, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss.SSSSSS');
                        catch
                            try
                                start_dt = datetime(start_str, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss.SSS');
                            catch
                                start_dt = datetime(start_str, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss');
                            end
                        end
                        duration_val = round(minutes(end_dt - start_dt), 1);
                        data.duration_min(row_idx) = duration_val;
                   
                    else
                        fprintf('Warning: start_str is empty or NaN string\n');
                    end
                else
                    fprintf('Warning: start_val is unexpected type: %s\n', class(start_val));
                end
            catch ME_dur
                fprintf('Warning: Could not calculate duration: %s\n', ME_dur.message);
            end
            
            if ~isempty(opts.n_trials)
                data.n_trials(row_idx) = opts.n_trials;
            end
            if ~isempty(opts.experiment_folder)
                if ~isstring(data.experiment_folder)
                    data.experiment_folder = strings(height(data), 1);
                end
                data.experiment_folder(row_idx) = string(opts.experiment_folder);
            end
            if ~isempty(opts.notes)
                if ~isstring(data.notes)
                    data.notes = strings(height(data), 1);
                end
                data.notes(row_idx) = string(opts.notes);
            end
            

            
            writetable(data, csv_path);
            fprintf('Updated task history: EMU-%d %s run-%d -> %s (duration: %.1f min)\n', ...
                    emu_num, task_name, run_num, opts.status, data.duration_min(row_idx));
        else
            % Row not found - log warning
            fprintf('Warning: Could not find entry to update: EMU-%d %s run-%d\n', ...
                    emu_num, task_name, run_num);
            fprintf('  Available entries in CSV: %d rows\n', height(data));
            if height(data) > 0
                fprintf('  EMU range: %d-%d, Tasks: %s\n', ...
                        min(data.emu_num), max(data.emu_num), ...
                        strjoin(unique(data.task), ', '));
            end
        end
        return
    end
    
    % Register new run if requested
    if opts.register
        new_row = table(...
            string(patient_id), ...
            emu_num, ...
            string(task_name), ...
            run_num, ...
            string(opts.status), ...
            string(datetime('now', 'Format', 'yyyy-MM-dd''T''HH:mm:ss.SSS')), ...
            "", ...  % end_time
            NaN, ... % duration_min
            NaN, ... % n_trials
            string(opts.experiment_folder), ...
            string(opts.notes), ...
            'VariableNames', headers);
        
        data = [data; new_row];
        writetable(data, csv_path);
        fprintf('Registered new run: EMU-%d %s run-%d\n', emu_num, task_name, run_num);
    end
end


function create_csv_from_folders(csv_path, headers, patient_id, scan_paths)
% Create a new CSV file, optionally scanning existing folders
    
    % Load existing data if CSV exists (for rescan/merge)
    if isfile(csv_path)
        try
            data = readtable(csv_path, 'TextType', 'string', 'Delimiter', ',');
            fprintf('Loaded existing CSV with %d entries\n', height(data));
        catch
            data = cell2table(cell(0, length(headers)), 'VariableNames', headers);
        end
    else
        % Create empty table with headers
        data = cell2table(cell(0, length(headers)), 'VariableNames', headers);
    end
    
    % Ensure all columns exist in data
    for i = 1:length(headers)
        if ~ismember(headers{i}, data.Properties.VariableNames)
            if ismember(headers{i}, {'emu_num', 'run_num', 'duration_min', 'n_trials'})
                data.(headers{i}) = nan(height(data), 1);
            else
                data.(headers{i}) = repmat("", height(data), 1);
            end
        end
    end
    
    % Scan paths for existing EMU folders if provided
    if ~isempty(scan_paths)
        fprintf('Scanning existing folders for task history...\n');
        
        for i = 1:length(scan_paths)
            scan_path = scan_paths{i};
            if isempty(scan_path) || ~isfolder(scan_path)
                continue
            end
            
            fprintf('  Scanning: %s\n', scan_path);
            
            % Look for EMU folders - try multiple patterns
            % Pattern 1: EMU folders with full naming convention
            emu_pattern1 = fullfile(scan_path, sprintf('EMU-*_subj-%s*', patient_id));
            emu_folders = dir(emu_pattern1);
            
            % Pattern 2: Any EMU folder (for folders that might not have subj- in name)
            emu_pattern2 = fullfile(scan_path, 'EMU-*');
            all_emu = dir(emu_pattern2);
            for k = 1:length(all_emu)
                if all_emu(k).isdir && ~any(strcmp({emu_folders.name}, all_emu(k).name))
                    emu_folders = [emu_folders; all_emu(k)];
                end
            end
            
            % Also check in transferred folder
            transferred_path = fullfile(scan_path, 'transferred');
            if isfolder(transferred_path)
                transferred_folders = dir(fullfile(transferred_path, sprintf('EMU-*_subj-%s*', patient_id)));
                emu_folders = [emu_folders; transferred_folders];
                % Also check any EMU folder in transferred
                all_transferred = dir(fullfile(transferred_path, 'EMU-*'));
                for k = 1:length(all_transferred)
                    if all_transferred(k).isdir && ~any(strcmp({emu_folders.name}, all_transferred(k).name))
                        emu_folders = [emu_folders; all_transferred(k)];
                    end
                end
            end
            
            % Also check in patient_id/EMU subfolder (if scan_path is backup_path)
            emu_subfolder = fullfile(scan_path, patient_id, 'EMU');
            if isfolder(emu_subfolder)
                emu_subfolders = dir(fullfile(emu_subfolder, 'EMU-*'));
                for k = 1:length(emu_subfolders)
                    if emu_subfolders(k).isdir && ~any(strcmp({emu_folders.name}, emu_subfolders(k).name))
                        emu_folders = [emu_folders; emu_subfolders(k)];
                    end
                end
            end
            
            fprintf('    Found %d potential EMU folders\n', length(emu_folders));
            
            for j = 1:length(emu_folders)
                folder = emu_folders(j);
                if ~folder.isdir
                    continue
                end
                
                % Parse folder name: EMU-XXX_subj-YYY_task-RSVPZZZ_run-WW
                % Try with subj- first
                tokens = regexp(folder.name, ...
                    'EMU-(\d+)_subj-([^_]+)_task-RSVP([^_]+)_run-(\d+)', 'tokens', 'once');
                
                % If no match, try without subj- requirement
                if isempty(tokens)
                    tokens = regexp(folder.name, ...
                        'EMU-(\d+).*_task-RSVP([^_]+)_run-(\d+)', 'tokens', 'once');
                    if ~isempty(tokens)
                        % Rearrange to match expected format
                        tokens = {tokens{1}, patient_id, tokens{2}, tokens{3}};
                    end
                end
                
                if isempty(tokens)
                    continue
                end
                
                emu_num = str2double(tokens{1});
                task_name = tokens{3};
                run_num = str2double(tokens{4});
                
                % Check if this entry already exists
                if height(data) > 0
                    existing = data.emu_num == emu_num & ...
                               strcmpi(data.task, task_name) & ...
                               data.run_num == run_num;
                    if any(existing)
                        continue
                    end
                end
                
                % Get folder creation time as approximate start time
                folder_path = fullfile(folder.folder, folder.name);
                folder_info = dir(folder_path);
                if ~isempty(folder_info)
                    start_time = datetime(folder.datenum, 'ConvertFrom', 'datenum', ...
                                          'Format', 'yyyy-MM-dd''T''HH:mm:ss.SSSSSS');
                else
                    start_time = "";
                end
                
                % Add to table
                new_row = table(...
                    string(patient_id), ...
                    emu_num, ...
                    string(task_name), ...
                    run_num, ...
                    "unknown", ...  % status
                    string(start_time), ...
                    "", ...  % end_time
                    NaN, ... % duration_min
                    NaN, ... % n_trials
                    string(folder_path), ...
                    "imported from folder", ...
                    'VariableNames', headers);
                
                data = [data; new_row];
                fprintf('  Found: EMU-%d %s run-%d\n', emu_num, task_name, run_num);
            end
            
            % Also scan for loose recording files (not in folders yet)
            % These are files like EMU-001_subj-XXX_task-RSVPDynamicScr_run-01.ns5
            loose_files = dir(fullfile(scan_path, 'EMU-*.ns*'));
            loose_files = [loose_files; dir(fullfile(scan_path, 'EMU-*.nev'))];
            
            for j = 1:length(loose_files)
                file = loose_files(j);
                if file.isdir
                    continue
                end
                
                % Parse filename: EMU-XXX_subj-YYY_task-RSVPZZZ_run-WW.ext
                [~, fname, ~] = fileparts(file.name);
                tokens = regexp(fname, ...
                    'EMU-(\d+)_subj-([^_]+)_task-RSVP([^_]+)_run-(\d+)', 'tokens', 'once');
                
                if isempty(tokens)
                    tokens = regexp(fname, ...
                        'EMU-(\d+).*_task-RSVP([^_]+)_run-(\d+)', 'tokens', 'once');
                    if ~isempty(tokens)
                        tokens = {tokens{1}, patient_id, tokens{2}, tokens{3}};
                    end
                end
                
                if isempty(tokens)
                    continue
                end
                
                emu_num_file = str2double(tokens{1});
                task_name_file = tokens{3};
                run_num_file = str2double(tokens{4});
                
                % Check if this entry already exists
                if height(data) > 0
                    existing = data.emu_num == emu_num_file & ...
                               strcmpi(data.task, task_name_file) & ...
                               data.run_num == run_num_file;
                    if any(existing)
                        continue
                    end
                end
                
                % Get file modification time as approximate start time
                start_time = datetime(file.datenum, 'ConvertFrom', 'datenum', ...
                                      'Format', 'yyyy-MM-dd''T''HH:mm:ss.SSSSSS');
                
                % Add to table (mark as loose files - no folder yet)
                new_row = table(...
                    string(patient_id), ...
                    emu_num_file, ...
                    string(task_name_file), ...
                    run_num_file, ...
                    "unknown", ...  % status
                    string(start_time), ...
                    "", ...  % end_time
                    NaN, ... % duration_min
                    NaN, ... % n_trials
                    string(file.folder), ...  % directory containing the file
                    "imported from loose file", ...
                    'VariableNames', headers);
                
                data = [data; new_row];
                fprintf('  Found loose file: EMU-%d %s run-%d\n', emu_num_file, task_name_file, run_num_file);
            end
        end
    end
    
    % Write the CSV (even if empty, creates the file with headers)
    writetable(data, csv_path);
    fprintf('Created Task History CSV: %s (%d entries)\n', csv_path, height(data));
end
