function success = wait_for_file_ready(data_file, timeout_sec, delete_ready_file)
    % WAIT_FOR_FILE_READY Waits for a .ready file to appear, indicating data is available
    %
    % This function waits for a companion .ready file that signals the data file
    % has been completely written by another MATLAB instance. Use in combination
    % with signal_file_ready() for explicit synchronization.
    %
    % Usage:
    %   success = wait_for_file_ready(data_file)
    %   success = wait_for_file_ready(data_file, timeout_sec)
    %   success = wait_for_file_ready(data_file, timeout_sec, delete_ready_file)
    %
    % Inputs:
    %   data_file         - Path to the data file to wait for
    %   timeout_sec       - Maximum time to wait in seconds (default: 60)
    %   delete_ready_file - Whether to delete the ready file after detection (default: true)
    %
    % Returns:
    %   success - true if ready file was detected, false if timeout occurred
    %
    % Example (Writer side):
    %   safe_save_mat('experiment.mat', '-struct', data);
    %   signal_file_ready('experiment.mat');
    %
    % Example (Reader side):
    %   if wait_for_file_ready('experiment.mat', 60)
    %       [data, ok] = safe_load_mat('experiment.mat');
    %   else
    %       error('Timeout waiting for experiment data');
    %   end
    %
    % See also: signal_file_ready, safe_load_mat
    
    if nargin < 2 || isempty(timeout_sec)
        timeout_sec = 60;
    end
    
    if nargin < 3 || isempty(delete_ready_file)
        delete_ready_file = true;
    end
    
    ready_file = [data_file '.ready'];
    success = false;
    
    tStart = tic;
    checkInterval = 0.2; % Check every 200ms
    
    fprintf('Waiting for ready signal: %s\n', ready_file);
    
    while toc(tStart) < timeout_sec
        if exist(ready_file, 'file')
            % Small delay to ensure file is fully written
            pause(0.1);
            
            % Verify the ready file is valid (not empty/corrupt)
            try
                fileInfo = dir(ready_file);
                if ~isempty(fileInfo) && fileInfo.bytes > 0
                    success = true;
                    
                    % Delete the ready file if requested
                    if delete_ready_file
                        try
                            delete(ready_file);
                        catch
                            warning('wait_for_file_ready: Could not delete ready file: %s', ready_file);
                        end
                    end
                    
                    fprintf('Ready signal received for: %s (waited %.1f sec)\n', data_file, toc(tStart));
                    return;
                end
            catch
                % File might have been deleted between exist check and dir
                % Continue waiting
            end
        end
        
        pause(checkInterval);
    end
    
    warning('wait_for_file_ready: Timeout after %.1f seconds waiting for: %s', timeout_sec, ready_file);
end
