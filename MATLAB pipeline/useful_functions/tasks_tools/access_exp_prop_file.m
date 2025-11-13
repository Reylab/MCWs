% function access_exp_prop_file(exp_prop_file, varName, varValue)
%     % Define backup filename
%     list_vars = ['experiment','scr_config_cell', 'pics_used_ids', 'scr_end_cell', 'abort', 'available_pics_cell', 'stim_rm_cell',
%         'stim_rm_max_trials_cell', 'selected2notremove_cell', 'priority_chs_ranking', 'selected2explore_cell', 'same_units_cell', 'same_categories_cell', 'picsexplored_names'];
%     [folder, name, ext] = fileparts(exp_prop_file);
%     backup_file = fullfile(folder, ['backup_' name ext]);
%     [~, backup_file_name,~] = fileparts(backup_file);
% 
%     % Max retry attempts
%     maxRetries = 5;
%     retries = 0;
% 
%     while retries < maxRetries
%         try
%             
%             % Try loading variable from the original file
%             if exist(exp_prop_file, 'file')
%                 try
%                     warnState = warning('off', 'MATLAB:load:variableNotFound');
%                     loadedVar = load(exp_prop_file, varName);
%                     %warning(warnState);
%                 catch
%                     error('access_exp_prop_file:CorruptOriginal', ...
%                         'Original file is corrupted: %s', exp_prop_file);
%                 end
%             else
%                 fprintf('Original file not found, creating new.\n');
%             end
% 
%             % Save variable to the original file
%             tmpStruct.(varName) = varValue;
%             if exist(exp_prop_file, 'file')
%                 save(exp_prop_file, '-struct', 'tmpStruct', '-append');
%             else
%                 save(exp_prop_file, '-struct', 'tmpStruct', '-v7');
%             end
% 
%             % Update backup after successful save
%             copyfile(exp_prop_file, backup_file);
%             break; % success
% 
%         catch ME
%             fprintf("\n");
%             fprintf(2, 'Error accessing %s: %s\n', exp_prop_file, ME.message);
%             retries = retries + 1;
% 
%             % Try restoring from backup
%             if exist(backup_file, 'file')
%                 try
%                     % Try loading backup
%                     load(backup_file);
%                     fprintf("\n");
%                     fprintf('Restoring from backup: %s\n', backup_file_name);
%                     delete(exp_prop_file);
%                     copyfile(backup_file, exp_prop_file);
%                     fprintf("\n");
%                     fprintf('Restored %s from backup.\n', exp_prop_file);
%                 catch
%                     fprintf("\n");
%                     fprintf(2, 'Backup file is also corrupted: %s\n', backup_file_name);
%                     fprintf("\n");
%                     fprintf('Creating fresh file.\n');
%                     tmpStruct.(varName) = varValue;
%                     save(exp_prop_file, '-struct', 'tmpStruct', '-v7');
%                     copyfile(exp_prop_file, backup_file);
%                 end
%             else
%                 fprintf('No backup found. Creating fresh file.\n');
%                 tmpStruct.(varName) = varValue;
%                 save(exp_prop_file, '-struct', 'tmpStruct', '-v7');
%                 copyfile(exp_prop_file, backup_file);
%             end
%         end
%     end
% 
%     if retries >= maxRetries
%         error('access_exp_prop_file:Failed', ...
%               'Both original and backup are corrupted for %s', exp_prop_file);
%     end
% end

% function access_exp_prop_file(exp_prop_file, varName, varValue)
% 
%     persistent lastVarName
%     
%     % Define backup filename
%     [folder, name, ext] = fileparts(exp_prop_file);
%     backup_file = fullfile(folder, ['backup_' name ext]);
%     [~, backup_file_name, ~] = fileparts(backup_file);
% 
%     % Max retry attempts
%     maxRetries = 5;
%     retries = 0;
% 
%     while retries < maxRetries
%         try
%             % Load all existing variables if the file exists
%             if exist(exp_prop_file, 'file')
%                 try
%                     warnState = warning('off', 'MATLAB:load:variableNotFound');
%                     allVars = load(exp_prop_file); % load everything
%                     %warning(warnState);
%                 catch
%                     fprintf(2, 'Error: Original file is corrupted: %s\n', exp_prop_file);
%                     if ~isempty(lastVarName)
%                         fprintf(2, 'Last variable attempted to save: %s\n', lastVarName);
%                     end
% 
%                     error('access_exp_prop_file:CorruptOriginal', ...
%                         'Original file is corrupted: %s', exp_prop_file);
%                     
%                 end
%             else
%                 fprintf('Original file not found, creating new.\n');
%                 allVars = struct();
%             end
% 
%             % Add/overwrite the requested variable
%             allVars.(varName) = varValue;
% 
%             % Save all variables back to file
%             save(exp_prop_file, '-struct', 'allVars', '-v7');
% 
%             % Update backup after successful save
%             copyfile(exp_prop_file, backup_file);
% 
%             % Update last saved variable name
%             lastVarName = varName;
% 
%             break; % success
% 
%         catch ME
%             fprintf("\n");
%             fprintf(2, 'Error accessing %s: %s\n', exp_prop_file, ME.message);
%             retries = retries + 1;
% 
%             % Try restoring from backup
%             if exist(backup_file, 'file')
%                 try
%                     allVars = load(backup_file); % load everything from backup
%                     fprintf("\n");
%                     fprintf('Restoring from backup: %s\n', backup_file_name);
% 
%                     delete(exp_prop_file);
%                     copyfile(backup_file, exp_prop_file);
% 
%                     fprintf("\n");
%                     fprintf('Restored %s from backup.\n', exp_prop_file);
%                 catch
%                     fprintf("\n");
%                     fprintf(2, 'Backup file is also corrupted: %s\n', backup_file_name);
%                     fprintf("\n");
%                     fprintf('Creating fresh file with all available variables.\n');
% 
%                     % start with empty struct if even backup is bad
%                     allVars = struct();
%                     allVars.(varName) = varValue;
%                     save(exp_prop_file, '-struct', 'allVars', '-v7');
%                     copyfile(exp_prop_file, backup_file);
%                 end
%             else
%                 fprintf('No backup found. Creating fresh file.\n');
%                 allVars = struct();
%                 allVars.(varName) = varValue;
%                 save(exp_prop_file, '-struct', 'allVars', '-v7');
%                 copyfile(exp_prop_file, backup_file);
%             end
%         end
%     end
% 
%     if retries >= maxRetries
%         error('access_exp_prop_file:Failed', ...
%               'Both original and backup are corrupted for %s', exp_prop_file);
%     end
% end
% 

function access_exp_prop_file(exp_prop_file, varName, varValue)

    persistent allVars lastVarName

    % Define backup filename
    [folder, name, ext] = fileparts(exp_prop_file);
    backup_file = fullfile(folder, ['backup_' name ext]);

    % Initialize persistent allVars if needed
    if isempty(allVars)
        if exist(exp_prop_file, 'file')
            try
                allVars = load(exp_prop_file);
            catch
                fprintf(2, 'Corrupted file: %s\n', exp_prop_file);
                if exist(backup_file, 'file')
                    fprintf('Trying backup...\n');
                    try
                        allVars = load(backup_file);
                    catch
                        fprintf(2, 'Backup also corrupted.\n');
                        allVars = struct(); % only because no in-memory exists yet
                    end
                else
                    fprintf('No backup found.\n');
                    allVars = struct(); % only because no in-memory exists yet
                end
            end
        else
            fprintf('No existing file found. Starting fresh.\n');
            allVars = struct();
        end
%     else
%         % Do nothing: keep the in-memory copy even if file/backup are bad
%         fprintf('Using persistent in-memory copy of allVars.\n');
    end

    % --- Update variable in memory ---
    allVars.(varName) = varValue;

    % --- Try saving to file and backup ---
    try
        save(exp_prop_file, '-struct', 'allVars', '-v7');
        copyfile(exp_prop_file, backup_file);
        lastVarName = varName;
    catch ME
        fprintf(2, 'Error saving %s: %s\n', exp_prop_file, ME.message);
        if ~isempty(lastVarName)
            fprintf(2, 'Last successfully saved variable: %s\n', lastVarName);
        end
        % but keep persistent allVars intact
    end
end

