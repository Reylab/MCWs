function status = check_tower_job_status(tower_ip, experiment_name, varargin)
% CHECK_TOWER_JOB_STATUS - Check the status of a processing job on Tower
%
% Reads the processing_status.txt file to determine job status.
% Supports hierarchical folder structure: Exp/sub_ID/EMU/taskID
%
% Usage:
%   status = check_tower_job_status(tower_ip, experiment_name)
%   status = check_tower_job_status(tower_ip, experiment_name, 'subject_id', 'MCW-FH_Test')
%
% Inputs:
%   tower_ip        - IP address of Tower computer
%   experiment_name - Name of the experiment (task folder name)
%
% Outputs:
%   status - Structure with fields:
%            .state     - 'pending', 'running', 'completed', 'failed', 'unknown'
%            .job_id    - SLURM job ID (if available)
%            .start_time - Start timestamp
%            .end_time   - End timestamp (if completed)
%            .message    - Status message

    p = inputParser;
    addParameter(p, 'username', 'user');
    addParameter(p, 'tower_exp_path', '/home/user/Documents/Exp');
    addParameter(p, 'subject_id', '');  % Subject ID for folder structure (e.g., MCW-FH_Test)
    parse(p, varargin{:});
    
    username = p.Results.username;
    tower_exp_path = p.Results.tower_exp_path;
    subject_id = p.Results.subject_id;
    
    ssh_conn = sprintf('%s@%s', username, tower_ip);
    
    % Build hierarchical path: Exp/sub_ID/EMU/taskID
    % Extract subject_id from experiment_name if not provided
    if isempty(subject_id)
        tokens = regexp(experiment_name, 'subj-([^_]+)', 'tokens');
        if ~isempty(tokens)
            subject_id = tokens{1}{1};
        end
    end
    
    % Build path based on available info
    % Structure: Exp/sub_ID/EMU/taskID (EMU is always just "EMU")
    if ~isempty(subject_id)
        status_file = sprintf('%s/%s/EMU/%s/processing_status.txt', tower_exp_path, subject_id, experiment_name);
    else
        % Fallback to flat structure
        status_file = sprintf('%s/%s/processing_status.txt', tower_exp_path, experiment_name);
    end
    
    % Initialize status structure
    status = struct();
    status.state = 'unknown';
    status.job_id = 0;
    status.start_time = '';
    status.end_time = '';
    status.message = '';
    status.experiment = experiment_name;
    status.status_file = status_file;
    
    % Check if status file exists and read it
    cat_cmd = sprintf('ssh %s "cat %s 2>/dev/null"', ssh_conn, status_file);
    [cmd_status, output] = system(cat_cmd);
    
    if cmd_status ~= 0 || isempty(output)
        status.state = 'pending';
        status.message = 'Job not started or status file not found';
        return;
    end
    
    % Parse status file
    lines = strsplit(output, newline);
    error_message = '';
    
    for i = 1:length(lines)
        line = strtrim(lines{i});
        
        if startsWith(line, 'STARTED:')
            status.state = 'running';
        elseif startsWith(line, 'COMPLETED:')
            status.state = 'completed';
        elseif startsWith(line, 'FAILED:')
            status.state = 'failed';
        elseif startsWith(line, 'Start time:')
            status.start_time = strtrim(strrep(line, 'Start time:', ''));
        elseif startsWith(line, 'End time:')
            status.end_time = strtrim(strrep(line, 'End time:', ''));
        elseif startsWith(line, 'Job ID:')
            job_str = strtrim(strrep(line, 'Job ID:', ''));
            status.job_id = str2double(job_str);
        elseif startsWith(line, 'Error:')
            error_message = strtrim(strrep(line, 'Error:', ''));
        end
    end
    
    % Build message
    switch status.state
        case 'pending'
            status.message = 'Job is pending';
        case 'running'
            status.message = sprintf('Job is running (started: %s)', status.start_time);
        case 'completed'
            status.message = sprintf('Job completed (started: %s, ended: %s)', ...
                status.start_time, status.end_time);
        case 'failed'
            if ~isempty(error_message)
                status.message = sprintf('Job failed: %s (started: %s, ended: %s)', ...
                    error_message, status.start_time, status.end_time);
            else
                status.message = sprintf('Job failed (started: %s, ended: %s). Check matlab_processing.log for details.', ...
                    status.start_time, status.end_time);
            end
        otherwise
            status.message = 'Unknown status';
    end
end
