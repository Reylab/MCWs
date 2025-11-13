function [status,msg] = move_file(source, destination, temp_folder)
    if exist("temp_folder", 'var')
        [status, msg] = copy_file(source, destination, temp_folder);
    else
        [status, msg] = copy_file(source, destination);
    end

    if status == 0
        if exist(source, 'file')
            try
                delete(source);
                fprintf('\nMoved %s to %s.\n', source, destination);
            catch ME
                warning('\nFailed to move %s to %s.\n', source, destination);
                errMsg = getReport(ME);
                disp(errMsg)
            end
        else
            warning(['\nFailed to move %s to %s. ' ...
                     'File not found or source is a dir.\n'], ...
                                            source, destination);
        end
    end
end