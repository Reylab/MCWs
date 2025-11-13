function [status,msg] = copy_file(source, destination, temp_folder)
    
    try
        [~,name,ext] = fileparts(source);
        if ~exist(destination,'dir')
            [status,msg] = mkdir(destination);
%             if ~isempty(msg)
%                 warning(msg)
%             end
        end
        if exist('temp_folder', 'var')
            if isunix
                [status,msg] = unix(sprintf('cp -r %s %s',source, temp_folder));
                
                [status,msg] = unix(sprintf('cp -r %s %s',[temp_folder filesep name ext], destination));
            else
                [status,msg] = copyfile(source, temp_folder);
                [status,msg] = copyfile([temp_folder filesep name ext], destination);
            end
        else
            if isunix
                [status,msg] = unix(sprintf('cp -r %s %s',source, destination));
            else
                [status,msg] = copyfile(source, destination);
            end
        end
%         if status == 0
%             fprintf('\n Copied %s to %s.\n', source, destination);
%         end
    catch ME
        fprintf('\n Copying %s to %s failed.\n', source, destination);
        errMsg = getReport(ME);
        disp(errMsg)
    end
end