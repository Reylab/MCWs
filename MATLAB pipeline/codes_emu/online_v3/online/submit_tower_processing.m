function [job_id, success] = submit_tower_processing(tower_ip, local_folder, experiment_name, varargin)
% SUBMIT_TOWER_PROCESSING - Copy data to Tower and run processing
%
% This function:
%   1. Copies the experiment folder to Tower with hierarchical structure:
%      Exp/sub_ID/EMU/taskID  (same as local backup structure)
%   2. Runs MATLAB processing via nohup with GPU rendering (vglrun)
%   3. Returns success status
%
% Usage:
%   [job_id, success] = submit_tower_processing(tower_ip, local_folder, experiment_name)
%   [job_id, success] = submit_tower_processing(..., 'subject_id', 'MCW-FH_Test')
%
% Inputs:
%   tower_ip        - IP address of Tower computer
%   local_folder    - Local path to experiment folder
%   experiment_name - Name of the experiment (taskID folder name)
%
% Outputs:
%   job_id  - -1 for nohup execution, 0 if failed
%   success - Boolean indicating if submission was successful

    p = inputParser;
    addParameter(p, 'username', 'user');
    addParameter(p, 'tower_exp_path', '/mnt/acq-hdd');
    addParameter(p, 'subject_id', '');  % Subject ID for folder structure (e.g., MCW-FH_Test)
    addParameter(p, 'is_online', true);
    addParameter(p, 'copy2miniscrfolder', true);
    addParameter(p, 'show_sel_count', true);
    addParameter(p, 'show_best_stims_wins', true);
    addParameter(p, 'max_spikes_plot', 500);
    addParameter(p, 'verbose', true);
    parse(p, varargin{:});
    
    username = p.Results.username;
    tower_exp_path = p.Results.tower_exp_path;
    subject_id = p.Results.subject_id;
    verbose = p.Results.verbose;
    
    job_id = 0;
    success = false;
    
    ssh_conn = sprintf('%s@%s', username, tower_ip);
    
    %% Step 1: Build hierarchical folder structure: Exp/subject_ID/EMU-XXX/task_folder
    % Try to extract subject_id from experiment_name if not provided
    if isempty(subject_id)
        % Pattern: EMU-XXX_subj-SUBJECT_task-TASK_run-X
        tokens = regexp(experiment_name, 'subj-([^_]+)', 'tokens');
        if ~isempty(tokens)
            subject_id = tokens{1}{1};
        else
            subject_id = 'unknown';
        end
    end
    
    % EMU folder is always just "EMU" (not EMU-XXX)
    % Structure: Exp/sub_ID/EMU/taskID
    emu_folder = 'EMU';
    remote_base_path = sprintf('%s/%s/%s', tower_exp_path, subject_id, emu_folder);
    
    if verbose
        fprintf('Creating remote directory structure on Tower...\n');
        fprintf('  Structure: %s/%s/%s/<task_folder>\n', tower_exp_path, subject_id, emu_folder);
    end
    
    mkdir_cmd = sprintf('ssh %s "mkdir -p %s"', ssh_conn, remote_base_path);
    [status, ~] = system(mkdir_cmd);
    if status ~= 0
        warning('Failed to create remote directory');
        return;
    end
    
    %% Step 2: Copy experiment folder to Tower using rsync
    if verbose
        fprintf('Copying experiment folder to Tower...\n');
        fprintf('  Source: %s\n', local_folder);
        fprintf('  Destination: %s:%s/\n', tower_ip, remote_base_path);
    end
    
    % Use rsync for efficient copying with progress to hierarchical path
    rsync_cmd = sprintf('rsync -avz --progress "%s" %s:%s/', ...
        local_folder, ssh_conn, remote_base_path);
    
    [status, output] = system(rsync_cmd);
    if status ~= 0
        warning('Failed to copy folder to Tower: %s', output);
        return;
    end
    
    if verbose
        fprintf('Copy completed successfully.\n');
    end
    
    %% Step 3: Run processing on Tower via nohup with GPU rendering (vglrun)
    remote_exp_folder = sprintf('%s/%s', remote_base_path, experiment_name);
    log_file = sprintf('%s/matlab_processing.log', remote_exp_folder);

    % Build MATLAB options string
    matlab_opts = sprintf('''is_online'', %s, ''copy2miniscrfolder'', %s, ''show_sel_count'', %s, ''show_best_stims_wins'', %s, ''max_spikes_plot'', %d', ...
        mat2str(p.Results.is_online), ...
        mat2str(p.Results.copy2miniscrfolder), ...
        mat2str(p.Results.show_sel_count), ...
        mat2str(p.Results.show_best_stims_wins), ...
        p.Results.max_spikes_plot);

    % Get the remote session DISPLAY so plots are visible on the active session
    [disp_status, remote_display] = system(sprintf('ssh %s "echo \\$DISPLAY"', ssh_conn));
    remote_display = strtrim(remote_display);
    if disp_status ~= 0 || isempty(remote_display)
        remote_display = ':10.0';  % Fallback
        if verbose
            fprintf('  Could not query DISPLAY from Tower, using fallback %s\n', remote_display);
        end
    else
        if verbose
            fprintf('  Tower DISPLAY: %s\n', remote_display);
        end
    end

    % Build processing script matching Tower framework:
    % Uses vglrun for GPU-accelerated rendering with NVIDIA offload
    % Explicitly set DISPLAY to the remote session so plots are visible
    script_content = sprintf([...
        '#!/bin/bash\n' ...
        'cd %s\n' ...
        'export DISPLAY=%s\n' ...
        'VGL_DISPLAY=/dev/dri/card0 __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia vglrun -d egl ' ...
        'matlab -nodesktop -nosplash -r "' ...
        'try; ' ...
        'addpath(''/home/user/Documents/GitHub/codes_emu/codes_for_analysis''); ' ...
        'processing_steps_MCW(%s); ' ...
        'fid=fopen(''processing_status.txt'',''a''); fprintf(fid,''COMPLETED\\n''); fprintf(fid,''End time: %%s\\n'',datestr(now)); fclose(fid); ' ...
        'catch ME; ' ...
        'fid=fopen(''processing_status.txt'',''a''); fprintf(fid,''FAILED\\n''); fprintf(fid,''Error: %%s\\n'',ME.message); fclose(fid); ' ...
        'disp(getReport(ME)); ' ...
        'end; ' ...
        'exit;"\n'], ...
        remote_exp_folder, ...
        remote_display, ...
        matlab_opts);

    % Write script locally and copy to Tower
    script_name = sprintf('run_processing_%s.sh', experiment_name);
    script_path = sprintf('%s/%s', remote_exp_folder, script_name);

    temp_script = tempname;
    fid = fopen(temp_script, 'w');
    fprintf(fid, '%s', script_content);
    fclose(fid);

    scp_cmd = sprintf('scp "%s" %s:%s', temp_script, ssh_conn, script_path);
    [status, ~] = system(scp_cmd);
    delete(temp_script);

    if status ~= 0
        warning('Failed to copy processing script to Tower');
        return;
    end

    system(sprintf('ssh %s "chmod +x %s"', ssh_conn, script_path));

    % Run with nohup in background
    if verbose
        fprintf('Starting processing on Tower with GPU rendering...\n');
    end

    run_cmd = sprintf('ssh %s "nohup bash %s > %s 2>&1 &"', ...
        ssh_conn, script_path, log_file);
    [status, output] = system(run_cmd);

    if status == 0
        job_id = -1;  % Indicate nohup execution
        success = true;
        if verbose
            fprintf('Processing started on Tower.\n');
            fprintf('Check progress: ssh %s "cat %s/processing_status.txt"\n', ssh_conn, remote_exp_folder);
        end
    else
        warning('Failed to start processing: %s', output);
    end
end
