% function [exp_prop_file_data] = load_exp_prop_file(exp_prop_file, varNames, allVars)
% 
%     try
%         % Try loading the requested variables from the MAT file
%         exp_prop_file_data = load(exp_prop_file, varNames{:});
%         
%         % If successful, return the loaded data
%         disp('File loaded successfully');
%         
%     catch ME
%         % If loading fails, just recreate the MAT file from scratch
%         fprintf(2, 'Error loading MAT file (%s). Recreating file.\n', ME.message);
% 
%         % Recreate fresh MAT file with the current structure
%         save(exp_prop_file, '-struct', 'allVars', varNames{:});
%         disp('Recreated fresh MAT file successfully.');
% 
%         % Return the updated structure (i.e., the allVars) after recreation
%         exp_prop_file_data = allVars;
%     end
% end

function load_exp_prop_file(exp_prop_file, varNames, allVarsFull, wait_for_ready, timeout_sec)
% LOAD_EXP_PROP_FILE Safely loads experiment properties with locking and ready signaling
%
% This function uses file locking and ready-file signaling to safely load
% data that may be written by another MATLAB instance.
%
% Usage:
%   load_exp_prop_file(exp_prop_file, varNames, allVarsFull)
%   load_exp_prop_file(exp_prop_file, varNames, allVarsFull, wait_for_ready)
%   load_exp_prop_file(exp_prop_file, varNames, allVarsFull, wait_for_ready, timeout_sec)
%
% Inputs:
%   exp_prop_file   - Experiment properties file path
%   varNames        - Cell array of variable names to load
%   allVarsFull     - Fallback struct if file is corrupted (for recreation)
%   wait_for_ready  - Whether to wait for ready signal (default: false)
%   timeout_sec     - Timeout in seconds for ready signal (default: 60)
%
% The loaded variables are assigned directly to the caller's workspace.
%
% See also: save_exp_prop_file, file_lock, wait_for_file_ready

    if nargin < 4 || isempty(wait_for_ready)
        wait_for_ready = false;
    end
    
    if nargin < 5 || isempty(timeout_sec)
        timeout_sec = 60;
    end
    
    maxRetries = 5;
    
    % Wait for ready signal if requested
    if wait_for_ready
        if ~wait_for_file_ready(exp_prop_file, timeout_sec)
            fprintf(2, 'Timeout waiting for ready signal: %s\n', exp_prop_file);
            fprintf(2, 'Falling back to allVarsFull data.\n');
            % Assign from fallback
            for k = 1:numel(varNames)
                if isfield(allVarsFull, varNames{k})
                    assignin('caller', varNames{k}, allVarsFull.(varNames{k}));
                else
                    assignin('caller', varNames{k}, []);
                end
            end
            return;
        end
    end
    
    % Acquire lock
    lock = file_lock(exp_prop_file, 30);
    if ~lock.acquire()
        fprintf(2, 'Could not acquire lock for %s. Using fallback data.\n', exp_prop_file);
        for k = 1:numel(varNames)
            if isfield(allVarsFull, varNames{k})
                assignin('caller', varNames{k}, allVarsFull.(varNames{k}));
            else
                assignin('caller', varNames{k}, []);
            end
        end
        return;
    end
    
    success = false;
    
    for attempt = 1:maxRetries
        try
            % Try loading the requested variables from the MAT file
            data = load(exp_prop_file, varNames{:});
            
            % Verify all variables were loaded
            allLoaded = true;
            for k = 1:numel(varNames)
                if ~isfield(data, varNames{k})
                    allLoaded = false;
                    break;
                end
            end
            
            if allLoaded
                % Assign each loaded variable to the caller workspace
                for k = 1:numel(varNames)
                    assignin('caller', varNames{k}, data.(varNames{k}));
                end
                success = true;
                break;
            else
                error('Not all requested variables found in file');
            end
            
        catch ME
            if attempt < maxRetries
                fprintf(2, 'Load attempt %d/%d failed: %s. Retrying...\n', ...
                    attempt, maxRetries, ME.message);
                pause(0.5);
            end
        end
    end
    
    lock.release();
    
    % If loading failed, recreate file and use fallback
    if ~success
        fprintf(2, 'Error loading MAT file after %d attempts. Recreating file.\n', maxRetries);
        
        try
            % Use save_exp_prop_file for safe recreation
            save_exp_prop_file(exp_prop_file, '', varNames, allVarsFull);
            fprintf('Recreated MAT file successfully.\n');
        catch
            % Direct save as last resort
            save(exp_prop_file, '-struct', 'allVarsFull', varNames{:});
        end
        
        % Assign all variables from allVarsFull to the caller workspace
        for k = 1:numel(varNames)
            if isfield(allVarsFull, varNames{k})
                assignin('caller', varNames{k}, allVarsFull.(varNames{k}));
            else
                assignin('caller', varNames{k}, []);
            end
        end
    end
end

