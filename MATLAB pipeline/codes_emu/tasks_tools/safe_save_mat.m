function success = safe_save_mat(filename, varargin)
    % SAFE_SAVE_MAT Safely saves variables to a MAT file with locking and atomic write
    %
    % This function provides safe file saving with:
    %   - File locking to prevent concurrent access
    %   - Atomic writes using temp file + rename
    %   - Automatic backup of previous file
    %   - Verification of saved data
    %
    % Usage:
    %   success = safe_save_mat(filename, 'var1', var1, 'var2', var2, ...)
    %   success = safe_save_mat(filename, '-struct', structVar)
    %   success = safe_save_mat(filename, '-struct', structVar, 'field1', 'field2', ...)
    %   success = safe_save_mat(filename, '-append', 'var1', var1, ...)
    %
    % Options:
    %   '-struct'  : Save fields of a structure as individual variables
    %   '-append'  : Append to existing file (loads existing, merges, saves)
    %
    % Returns:
    %   success - true if save was successful, false otherwise
    %
    % Example:
    %   data.x = 1:10;
    %   data.y = rand(10,1);
    %   success = safe_save_mat('mydata.mat', '-struct', data);
    %
    % See also: safe_load_mat, file_lock, signal_file_ready
    
    success = false;
    
    if isempty(varargin)
        warning('safe_save_mat: No variables specified to save');
        return;
    end
    
    % Acquire lock
    lock = file_lock(filename, 30);
    if ~lock.acquire()
        warning('safe_save_mat: Could not acquire lock for %s', filename);
        return;
    end
    
    try
        % Parse options
        appendMode = false;
        structMode = false;
        structVar = [];
        varNames = {};
        varValues = {};
        
        idx = 1;
        while idx <= length(varargin)
            arg = varargin{idx};
            
            if ischar(arg) && strcmp(arg, '-append')
                appendMode = true;
                idx = idx + 1;
            elseif ischar(arg) && strcmp(arg, '-struct')
                structMode = true;
                idx = idx + 1;
                if idx <= length(varargin)
                    structVar = varargin{idx};
                    idx = idx + 1;
                    % Remaining args are field names to save
                    while idx <= length(varargin)
                        varNames{end+1} = varargin{idx}; %#ok<AGROW>
                        idx = idx + 1;
                    end
                end
            else
                % Name-value pair
                if idx + 1 <= length(varargin)
                    varNames{end+1} = arg; %#ok<AGROW>
                    varValues{end+1} = varargin{idx + 1}; %#ok<AGROW>
                    idx = idx + 2;
                else
                    warning('safe_save_mat: Unmatched variable name: %s', arg);
                    idx = idx + 1;
                end
            end
        end
        
        % Build save structure
        if structMode
            if isempty(varNames)
                % Save all fields
                saveStruct = structVar;
            else
                % Save only specified fields
                saveStruct = struct();
                for i = 1:length(varNames)
                    if isfield(structVar, varNames{i})
                        saveStruct.(varNames{i}) = structVar.(varNames{i});
                    else
                        warning('safe_save_mat: Field not found in struct: %s', varNames{i});
                    end
                end
            end
        else
            % Build struct from name-value pairs
            saveStruct = struct();
            for i = 1:length(varNames)
                saveStruct.(varNames{i}) = varValues{i};
            end
        end
        
        % Handle append mode - load existing and merge
        if appendMode && exist(filename, 'file')
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
                warning('safe_save_mat: Could not load existing file for append: %s', ME.message);
            end
        end
        
        % Create temp file path
        [filePath, fileName, fileExt] = fileparts(filename);
        if isempty(filePath)
            filePath = pwd;
        end
        temp_file = fullfile(filePath, [fileName '_temp_' datestr(now, 'HHMMSS') fileExt]);
        
        % Save to temp file
        save(temp_file, '-struct', 'saveStruct', '-v7.3');
        
        % Verify temp file is valid and readable
        try
            testLoad = load(temp_file);
            savedFields = fieldnames(saveStruct);
            for i = 1:length(savedFields)
                if ~isfield(testLoad, savedFields{i})
                    error('Verification failed: field %s not saved correctly', savedFields{i});
                end
            end
            clear testLoad;
        catch ME
            if exist(temp_file, 'file')
                delete(temp_file);
            end
            error('Verification of saved file failed: %s', ME.message);
        end
        
        % Create backup of existing file
        if exist(filename, 'file')
            backup_file = fullfile(filePath, [fileName '_backup' fileExt]);
            try
                copyfile(filename, backup_file);
            catch
                warning('safe_save_mat: Could not create backup file');
            end
        end
        
        % Atomic move: replace original with temp
        movefile(temp_file, filename);
        
        success = true;
        
    catch ME
        warning('safe_save_mat: Save failed: %s', ME.message);
        % Clean up temp file if it exists
        if exist('temp_file', 'var') && exist(temp_file, 'file')
            try
                delete(temp_file);
            catch
            end
        end
    end
    
    lock.release();
end
