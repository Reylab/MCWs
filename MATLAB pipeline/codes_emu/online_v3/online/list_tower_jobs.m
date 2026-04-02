function jobs = list_tower_jobs(varargin)
% LIST_TOWER_JOBS - List all processing jobs on Tower and their status
%
% Scans the Tower's hierarchical directory structure:
%   Exp/subject_id/EMU-XXX/task_folder
% and shows the status of each processing job.
%
% Usage:
%   jobs = list_tower_jobs()
%   jobs = list_tower_jobs('subject_id', 'MCW-FH')  % Filter by subject
%   jobs = list_tower_jobs('tower_ip', '10.238.92.106')
%
% Outputs:
%   jobs - Table with columns: Subject, EMU, Experiment, Status, JobID, StartTime, EndTime

    p = inputParser;
    addParameter(p, 'tower_ip', '');  % Auto-detect if empty
    addParameter(p, 'username', 'user');
    addParameter(p, 'tower_exp_path', '/home/user/Documents/Exp');
    addParameter(p, 'subject_id', '');  % Filter by subject (empty = all subjects)
    addParameter(p, 'verbose', true);
    parse(p, varargin{:});
    
    tower_ip = p.Results.tower_ip;
    username = p.Results.username;
    tower_exp_path = p.Results.tower_exp_path;
    subject_filter = p.Results.subject_id;
    verbose = p.Results.verbose;
    
    %% Connect to Tower if IP not provided
    if isempty(tower_ip)
        [success, tower_ip, ~] = connect_to_tower('verbose', verbose);
        if ~success
            error('Could not connect to Tower. Please specify tower_ip manually.');
        end
    end
    
    ssh_conn = sprintf('%s@%s', username, tower_ip);
    
    if verbose
        fprintf('\n=== Tower Processing Jobs ===\n');
        fprintf('Tower: %s\n', tower_ip);
        fprintf('Path: %s\n\n', tower_exp_path);
    end
    
    %% List subjects (first level directories)
    if ~isempty(subject_filter)
        subjects = {subject_filter};
    else
        list_cmd = sprintf('ssh %s "ls -1 %s 2>/dev/null"', ssh_conn, tower_exp_path);
        [status, output] = system(list_cmd);
        
        if status ~= 0 || isempty(strtrim(output))
            if verbose
                fprintf('No subjects found on Tower.\n');
            end
            jobs = table();
            return;
        end
        
        subjects = strsplit(strtrim(output), newline);
        subjects = subjects(~cellfun(@isempty, subjects));
    end
    
    %% Get status for each experiment (hierarchical search: Exp/sub_ID/EMU/taskID)
    subjects_list = {};
    experiments = {};
    statuses = {};
    job_ids = {};
    start_times = {};
    end_times = {};
    
    for si = 1:length(subjects)
        subject_id = subjects{si};
        emu_path = sprintf('%s/%s/EMU', tower_exp_path, subject_id);
        
        % List task folders in this subject's EMU folder
        list_task_cmd = sprintf('ssh %s "ls -1 %s 2>/dev/null"', ssh_conn, emu_path);
        [status, task_output] = system(list_task_cmd);
        
        if status ~= 0 || isempty(strtrim(task_output))
            continue;  % No task folders for this subject
        end
        
        tasks = strsplit(strtrim(task_output), newline);
        tasks = tasks(~cellfun(@isempty, tasks));
        
        for ti = 1:length(tasks)
            exp_name = tasks{ti};
            
            % Check status with hierarchical path info
            job_status = check_tower_job_status(tower_ip, exp_name, ...
                'username', username, 'tower_exp_path', tower_exp_path, ...
                'subject_id', subject_id);
            
            subjects_list{end+1} = subject_id;
            experiments{end+1} = exp_name;
            statuses{end+1} = job_status.state;
            job_ids{end+1} = job_status.job_id;
            start_times{end+1} = job_status.start_time;
            end_times{end+1} = job_status.end_time;
        end
    end
    
    %% Create output table
    if isempty(experiments)
        if verbose
            fprintf('No experiments found on Tower.\n');
        end
        jobs = table();
        return;
    end
    
    jobs = table(subjects_list', experiments', statuses', cell2mat(job_ids)', start_times', end_times', ...
        'VariableNames', {'Subject', 'Experiment', 'Status', 'JobID', 'StartTime', 'EndTime'});
    
    %% Display if verbose
    if verbose
        if isempty(jobs)
            fprintf('No experiments found.\n');
        else
            % Count by status
            n_pending = sum(strcmp(jobs.Status, 'pending'));
            n_running = sum(strcmp(jobs.Status, 'running'));
            n_completed = sum(strcmp(jobs.Status, 'completed'));
            n_failed = sum(strcmp(jobs.Status, 'failed'));
            
            fprintf('Summary: %d pending, %d running, %d completed, %d failed\n\n', ...
                n_pending, n_running, n_completed, n_failed);
            
            % Print table header (structure: Exp/sub_ID/EMU/taskID)
            fprintf('%-20s %-45s %-15s %-20s\n', 'Subject', 'Experiment', 'Status', 'Start Time');
            fprintf('%s\n', repmat('-', 1, 105));
            
            for i = 1:height(jobs)
                status_str = jobs.Status{i};
                switch status_str
                    case 'completed'
                        status_disp = sprintf('[OK] %s', status_str);
                    case 'failed'
                        status_disp = sprintf('[!!] %s', status_str);
                    case 'running'
                        status_disp = sprintf('[>>] %s', status_str);
                    otherwise
                        status_disp = sprintf('[..] %s', status_str);
                end
                
                fprintf('%-20s %-45s %-15s %-20s\n', ...
                    jobs.Subject{i}, jobs.Experiment{i}, status_disp, jobs.StartTime{i});
            end
            fprintf('\n');
        end
    end
end
