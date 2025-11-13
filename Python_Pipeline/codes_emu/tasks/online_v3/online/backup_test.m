function backup_test

experiment.folder_name = 'EMU-009_subj-MCW-FH_005_task-RSVPdynamic_scr_test_run-02';
params.acq_remote_folder_in_processing = '/media/acq';
params.recording_name = 'EMU-009_subj-MCW-FH_005_task-RSVPdynamic_scr_test_run-02_RIP';
params.backup_path = '/mnt/acq-hdd';
params.sub_ID = 'MCW-FH_005';
params.root_processing = '/home/user/share/experimental_files/';
params.with_acq_folder=true;

% addpath(fullfile(fileparts(mfilename('fullpath')),'../../..')) % always / for this trick
% custompath = reylab_custompath({'useful_functions','wave_clus_reylab','wave_split','codes_for_analysis','mex','tasks/online_v3/online'});

disp('Starting backup worker...')
pause(3)
backup_worker = parfeval(@backup_raw_data,1,params,experiment.folder_name);

disp('Copping data from beh...')
%     [status,msg] = copyfile([params.root_processing filesep experiment.folder_name], [params.backup_path filesep params.sub_ID filesep experiment.folder_name]);
if ~isunix
%     [~,msg] = copyfile(fullfile(params.root_processing,experiment.folder_name), fullfile(params.backup_path ,params.sub_ID,experiment.folder_name),'-a');
    [~,msg] = copyfile(fullfile(params.root_processing,experiment.folder_name), fullfile(params.backup_path ,params.sub_ID),'-a');
else
%     [~,msg] = unix(sprintf('cp -r %s %s',fullfile(params.root_processing,experiment.folder_name), fullfile(params.backup_path ,params.sub_ID,experiment.folder_name)));
    [~,msg] = unix(sprintf('cp -r %s %s',fullfile(params.root_processing,experiment.folder_name), fullfile(params.backup_path ,params.sub_ID)));
end
if ~isempty(msg)
    warning('Error copying files: %s', msg)
end
if params.with_acq_folder
    if exist('backup_worker','var') && ~strcmp(backup_worker.State, 'finished')
        disp('wait, copying raw files to backup...')
        wait(backup_worker)
    end
    if exist('backup_worker','var') && isfield(backup_worker,'Error') && ~isempty(backup_worker.Error.message)
        warning("backup worker error: %s\n", backup_worker.Error.message)
    else
        [bw_msg] = fetchOutputs(backup_worker);
        if ~isempty(bw_msg)
            warning('Error copying files: %s', bw_msg)
        else
            disp('raw files backup done.')
        end
    end
end
end
    function msg = backup_raw_data(params, folder_name)
        [~,msg] = mkdir(fullfile(params.acq_remote_folder_in_processing, folder_name));
    
        if ~isempty(msg), return; end
        if ~isunix
            [~,msg] = movefile(fullfile(params.acq_remote_folder_in_processing,[params.recording_name '*']),fullfile(params.acq_remote_folder_in_processing,folder_name));
        else
            [~,msg] = unix(sprintf('mv %s %s',fullfile(params.acq_remote_folder_in_processing,[params.recording_name '*']), fullfile(params.acq_remote_folder_in_processing,folder_name)));
        end
        if ~isempty(msg), return; end
    
        if ~isunix
%             [~,msg] = copyfile(fullfile(params.acq_remote_folder_in_processing,folder_name), fullfile(params.backup_path,params.sub_ID,folder_name));
            [~,msg] = copyfile(fullfile(params.acq_remote_folder_in_processing,folder_name), fullfile(params.backup_path,params.sub_ID));
        else
%             [~,msg] = unix(sprintf('cp -r %s %s',fullfile(params.acq_remote_folder_in_processing,folder_name), fullfile(params.backup_path,params.sub_ID,folder_name)));
            [~,msg] = unix(sprintf('cp -r %s %s',fullfile(params.acq_remote_folder_in_processing,folder_name), fullfile(params.backup_path,params.sub_ID)));
        end
        if ~isempty(msg), return; end
    
        if ~isunix
%             [~,msg] =  movefile(fullfile(params.acq_remote_folder_in_processing,folder_name),fullfile(params.acq_remote_folder_in_processing,'transferred',folder_name));
            [~,msg] =  movefile(fullfile(params.acq_remote_folder_in_processing,folder_name),fullfile(params.acq_remote_folder_in_processing,'transferred'));
        else
%             [~,msg]  = unix(sprintf('mv %s %s',fullfile(params.acq_remote_folder_in_processing,folder_name),fullfile(params.acq_remote_folder_in_processing,'transferred',folder_name)));
            [~,msg]  = unix(sprintf('mv %s %s',fullfile(params.acq_remote_folder_in_processing,folder_name),fullfile(params.acq_remote_folder_in_processing,'transferred')));
        end
    end

