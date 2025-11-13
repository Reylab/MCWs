function [status, msg] = copy_file(source, destination, temp_folder)
    try
        [~, name, ext] = fileparts(source);
        
        % Check if source is a folder
        isSourceFolder = isfolder(source);
        
        % Ensure destination exists
        if ~exist(destination, 'dir')
            [status, msg] = mkdir(destination);
        else
            status = 0; msg = '';
        end

        if isSourceFolder
            % ----------- FOLDER COPY -----------
            target_folder = fullfile(destination, name);
            if ~exist(target_folder, 'dir')
                mkdir(target_folder);
            end

            if exist('temp_folder', 'var')
                if isunix
                    % Copy to temp folder first
                    [status, msg] = unix(sprintf('cp -r "%s" "%s"', source, temp_folder));
                    % Copy contents of temp folder to target folder (avoid nesting)
                    [status, msg] = unix(sprintf('cp -r "%s/." "%s"', ...
                        fullfile(temp_folder, name, ext), target_folder));
                else
                    [status, msg] = copyfile(source, temp_folder);
                    [status, msg] = copyfile([temp_folder filesep name ext filesep '*'], target_folder);
                end
            else
                if isunix
                    [status, msg] = unix(sprintf('cp -r "%s/." "%s"', source, target_folder));
                else
                    [status, msg] = copyfile([source filesep '*'], target_folder);
                end
            end

        else
            % ----------- FILE COPY -----------
            if exist('temp_folder', 'var')
                if isunix
                    [status, msg] = unix(sprintf('cp -r "%s" "%s"', source, temp_folder));
                    [status, msg] = unix(sprintf('cp -r "%s" "%s"', ...
                        fullfile(temp_folder, [name ext]), destination));
                else
                    [status, msg] = copyfile(source, temp_folder);
                    [status, msg] = copyfile(fullfile(temp_folder, [name ext]), destination);
                end
            else
                if isunix
                    [status, msg] = unix(sprintf('cp -r "%s" "%s"', source, destination));
                else
                    [status, msg] = copyfile(source, destination);
                end
            end
        end

    catch ME
        fprintf('\n Copying %s to %s failed.\n', source, destination);
        errMsg = getReport(ME);
        disp(errMsg)
    end
end
