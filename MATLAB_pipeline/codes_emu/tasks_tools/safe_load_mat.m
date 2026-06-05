function [data, success] = safe_load_mat(filename, varargin)
    % SAFE_LOAD_MAT Safely loads variables from a MAT file with locking and retry
    %
    % This function provides safe file loading with:
    %   - File locking to prevent reading during write
    %   - Automatic retry on failure
    %   - Verification of loaded variables
    %
    % Usage:
    %   [data, success] = safe_load_mat(filename)
    %   [data, success] = safe_load_mat(filename, 'var1', 'var2', ...)
    %   [data, success] = safe_load_mat(filename, 'var1', 'var2', ..., 'MaxRetries', 5)
    %   [data, success] = safe_load_mat(filename, 'var1', 'var2', ..., 'RetryDelay', 1)
    %
    % Options:
    %   'MaxRetries'  : Maximum number of load attempts (default: 5)
    %   'RetryDelay'  : Delay between retries in seconds (default: 1)
    %
    % Returns:
    %   data    - Struct containing loaded variables
    %   success - true if load was successful, false otherwise
    %
    % Example:
    %   [data, ok] = safe_load_mat('mydata.mat', 'x', 'y');
    %   if ok
    %       disp(data.x);
    %   end
    %
    % See also: safe_save_mat, file_lock, wait_for_file_ready
    
    data = struct();
    success = false;
    
    % Parse options
    maxRetries = 5;
    retryDelay = 1;
    varNames = {};
    
    idx = 1;
    while idx <= length(varargin)
        arg = varargin{idx};
        if ischar(arg)
            if strcmpi(arg, 'MaxRetries') && idx + 1 <= length(varargin)
                maxRetries = varargin{idx + 1};
                idx = idx + 2;
            elseif strcmpi(arg, 'RetryDelay') && idx + 1 <= length(varargin)
                retryDelay = varargin{idx + 1};
                idx = idx + 2;
            else
                % It's a variable name
                varNames{end + 1} = arg; %#ok<AGROW>
                idx = idx + 1;
            end
        else
            idx = idx + 1;
        end
    end
    
    % Check if file exists
    if ~exist(filename, 'file')
        warning('safe_load_mat: File does not exist: %s', filename);
        return;
    end
    
    % Acquire lock
    lock = file_lock(filename, 30);
    if ~lock.acquire()
        warning('safe_load_mat: Could not acquire lock for %s', filename);
        return;
    end
    
    try
        for attempt = 1:maxRetries
            try
                % Load file
                if isempty(varNames)
                    data = load(filename);
                else
                    data = load(filename, varNames{:});
                end
                
                % Verify all requested variables were loaded
                if ~isempty(varNames)
                    missingVars = {};
                    for i = 1:length(varNames)
                        if ~isfield(data, varNames{i})
                            missingVars{end + 1} = varNames{i}; %#ok<AGROW>
                        end
                    end
                    if ~isempty(missingVars)
                        error('Missing variables: %s', strjoin(missingVars, ', '));
                    end
                end
                
                success = true;
                break;
                
            catch ME
                if attempt < maxRetries
                    warning('safe_load_mat: Load attempt %d/%d failed: %s. Retrying in %.1f sec...', ...
                        attempt, maxRetries, ME.message, retryDelay);
                    pause(retryDelay);
                else
                    warning('safe_load_mat: All %d load attempts failed. Last error: %s', ...
                        maxRetries, ME.message);
                end
            end
        end
        
    catch ME
        warning('safe_load_mat: Unexpected error: %s', ME.message);
    end
    
    lock.release();
end
