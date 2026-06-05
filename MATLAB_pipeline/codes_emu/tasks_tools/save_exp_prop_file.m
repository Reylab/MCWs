function save_exp_prop_file(exp_prop_file, subscr_exp_prop_file, varNames, allVars)
% SAVE_EXP_PROP_FILE Safely saves experiment properties with locking and ready signaling
%
% This function uses file locking and atomic writes to prevent corruption
% when multiple MATLAB instances access the same file.
%
% Usage:
%   save_exp_prop_file(exp_prop_file, subscr_exp_prop_file, varNames, allVars)
%
% Inputs:
%   exp_prop_file       - Primary experiment properties file path
%   subscr_exp_prop_file - Secondary (subscreening) file path, or '' to skip
%   varNames            - Cell array of variable names to save
%   allVars             - Struct containing the variables to save
%
% The function:
%   1. Acquires a file lock to prevent concurrent access
%   2. Loads existing data (for append behavior)
%   3. Merges new data with existing
%   4. Saves to a temp file first
%   5. Verifies the temp file
%   6. Atomically moves temp to target
%   7. Signals ready for readers
%   8. Releases the lock
%
% See also: safe_save_mat, safe_load_mat, file_lock, signal_file_ready

    maxRetries = 5;
    
    % Build struct with only requested variables
    saveStruct = struct();
    for i = 1:length(varNames)
        if isfield(allVars, varNames{i})
            saveStruct.(varNames{i}) = allVars.(varNames{i});
        end
    end
    
    % Save primary file
    success = save_with_lock_and_signal(exp_prop_file, saveStruct, maxRetries);
    if ~success
        error('Failed to save %s after %d attempts.', exp_prop_file, maxRetries);
    end
    
    % Save secondary file if provided
    if ~isempty(subscr_exp_prop_file)
        success = save_with_lock_and_signal(subscr_exp_prop_file, saveStruct, maxRetries);
        if ~success
            error('Failed to save %s after %d attempts.', subscr_exp_prop_file, maxRetries);
        end
    end
end


function success = save_with_lock_and_signal(filename, saveStruct, maxRetries)
% Internal function to save with locking, atomic write, and ready signaling

    success = false;
    
    for attempt = 1:maxRetries
        % Acquire lock
        lock = file_lock(filename, 30);
        if ~lock.acquire()
            fprintf(2, 'Warning: Could not acquire lock for %s (attempt %d/%d)\n', ...
                filename, attempt, maxRetries);
            pause(0.5);
            continue;
        end
        
        try
            % Load existing data for merge (append behavior)
            if exist(filename, 'file')
                try
                    existingData = load(filename);
                    % Merge: new data overwrites existing
                    existingFields = fieldnames(existingData);
                    for i = 1:length(existingFields)
                        if ~isfield(saveStruct, existingFields{i})
                            saveStruct.(existingFields{i}) = existingData.(existingFields{i});
                        end
                    end
                catch ME
                    fprintf(2, 'Warning: Could not load existing file %s: %s\n', filename, ME.message);
                    % Continue with just the new data
                end
            end
            
            % Create temp file
            [filePath, fileName, fileExt] = fileparts(filename);
            if isempty(filePath)
                filePath = pwd;
            end
            temp_file = fullfile(filePath, [fileName '_temp_' datestr(now, 'HHMMSS_FFF') fileExt]);
            
            % Save to temp file
            save(temp_file, '-struct', 'saveStruct', '-v7.3');
            
            % Verify temp file is valid
            try
                testLoad = load(temp_file);
                clear testLoad;
            catch ME
                if exist(temp_file, 'file')
                    delete(temp_file);
                end
                error('Temp file verification failed: %s', ME.message);
            end
            
            % Create backup of existing file
            if exist(filename, 'file')
                backup_file = fullfile(filePath, [fileName '_backup' fileExt]);
                try
                    copyfile(filename, backup_file);
                catch
                    % Backup failed, but continue anyway
                end
            end
            
            % Atomic move: replace original with temp
            movefile(temp_file, filename);
            
            % Release lock before signaling (so reader can acquire lock)
            lock.release();
            
            % Signal that file is ready for reading
            signal_file_ready(filename);
            
            success = true;
            return;
            
        catch ME
            fprintf(2, 'Warning: Save attempt %d/%d failed for %s: %s\n', ...
                attempt, maxRetries, filename, ME.message);
            
            % Clean up temp file if it exists
            if exist('temp_file', 'var') && exist(temp_file, 'file')
                try
                    delete(temp_file);
                catch
                end
            end
            
            lock.release();
            pause(0.2);
        end
    end
end
