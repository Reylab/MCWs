function [msg] = backup_raw_data(params, folder_name)

    if strcmp(params.location,'MCW-BEH-RIP') || contains(params.location, 'ABT')
        acq_remote_folder = params.acq_remote_folder_in_beh;
        [status,msg] = mkdir(fullfile(acq_remote_folder, folder_name));
        mkdir(fullfile(params.backup_path,params.sub_ID,'EMU'))
    elseif strcmp(params.location,'MCW-FH-RIP')
        acq_remote_folder = params.acq_remote_folder_in_processing;
        [status,msg] = mkdir(fullfile(params.acq_remote_folder_in_processing, folder_name));
        mkdir(fullfile(params.backup_path,params.sub_ID,'EMU'))
    end

    if ~status, return; end
    
    %if ~isempty(msg), return; end
    if ~isunix
        [~,msg] = movefile(fullfile(acq_remote_folder,[params.recording_name '*']),fullfile(acq_remote_folder,folder_name));
    else
        [~,msg] = unix(sprintf('mv %s %s',fullfile(acq_remote_folder,[params.recording_name '*']), fullfile(acq_remote_folder,folder_name)));
    end
    if ~isempty(msg), return; end

    if ~exist(fullfile(params.backup_path, params.sub_ID, 'EMU', folder_name), 'dir')
        mkdir(fullfile(params.backup_path, params.sub_ID, 'EMU', folder_name));
    end
    
    if ~isunix
        %             [~,msg] = copyfile(fullfile(params.acq_remote_folder_in_processing,folder_name), fullfile(params.backup_path,params.sub_ID,folder_name));
        %[~,msg] = copyfile(fullfile(acq_remote_folder,folder_name), fullfile(params.backup_path,params.sub_ID,'EMU'));
        % Copy all contents into the backup folder on Windows
        [~, msg] = copyfile(fullfile(acq_remote_folder, folder_name, '*'), ...
                        fullfile(params.backup_path, params.sub_ID, 'EMU', folder_name));
    else
        %             [~,msg] = unix(sprintf('cp -r %s %s',fullfile(params.acq_remote_folder_in_processing,folder_name), fullfile(params.backup_path,params.sub_ID,folder_name)));
        %[~,msg] = unix(sprintf('cp -r %s %s',fullfile(acq_remote_folder,folder_name), fullfile(params.backup_path,params.sub_ID,'EMU',folder_name)));
        % Copy all contents (including hidden files) into the backup folder on Unix
        [~, msg] = unix(sprintf('cp -r "%s"/. "%s"', ...
                            fullfile(acq_remote_folder, folder_name), ...
                            fullfile(params.backup_path, params.sub_ID, 'EMU', folder_name)));
    end
    if ~isempty(msg), return; end
    
    if ~isunix
        %             [~,msg] =  movefile(fullfile(params.acq_remote_folder_in_processing,folder_name),fullfile(params.acq_remote_folder_in_processing,'transferred',folder_name));
        [~,msg] =  movefile(fullfile(acq_remote_folder,folder_name),fullfile(acq_remote_folder,'transferred'));
    else
        %             [~,msg]  = unix(sprintf('mv %s %s',fullfile(params.acq_remote_folder_in_processing,folder_name),fullfile(params.acq_remote_folder_in_processing,'transferred',folder_name)));
        [~,msg]  = unix(sprintf('mv %s %s',fullfile(acq_remote_folder,folder_name),fullfile(acq_remote_folder,'transferred')));
    end
end