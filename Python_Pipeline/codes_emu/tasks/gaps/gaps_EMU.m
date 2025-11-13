function gaps_EMU(varargin)
% gaps_EMU('mode', 'test')
%this code should be run in the behaviour computer
close all
clc

ipr = inputParser;
addOptional(ipr,'sub_ID', []); %if empty it will check current_subject_id.txt to get subject ID
addOptional(ipr,'location', 'MCW-FH-RIP'); % in this task, same if using 'MCW-BEH-RIP'
addOptional(ipr,'mode', 'real'); % in this task, same if using 'MCW-BEH-RIP'
addParameter(ipr,'EMU_num',[]) %numeric value, if empty check files in acq folder and logs to automatically find the next one.
addParameter(ipr,'with_acq_folder',[]) % after finishing move the files inside folder, requires access to folder with raw files
parse(ipr,varargin{:})

if strcmp(ipr.Results.mode,'real')
    rec_length = 3600; % time in sec before splitting a recording. Set to 3600 (1 hour)
elseif strcmp(ipr.Results.mode,'test')
    rec_length = 60; % time in sec before splitting a recording. 
end

addpath(fullfile(fileparts(mfilename('fullpath')),'../..')) % always / for this trick
custompath = reylab_custompath({'useful_functions','tasks/.'});

params = location_setup(ipr.Results.location);
inputs = fields(ipr.Results);
for i =1:numel(inputs)
    pstr = inputs{i};
    if any(strcmp(ipr.UsingDefaults,pstr)) && isempty(ipr.Results.(pstr)) && isfield(params, pstr) %parameter not passed and empty
        continue
    end
    params.(pstr) = ipr.Results.(pstr);
end

if isempty(params.sub_ID)
    params.sub_ID = strtrim(fileread(fullfile(params.beh_rec_metadata,'current_subject_id.txt')));
end

if strcmp(params.system, 'RIP')
    custompath.add(params.xippmex_path,true)
elseif strcmp(params.system, 'BRK')
    custompath.add(params.cbmex_path,true)
else
    custompath.rm()
    error('unknown system');
end

if params.with_acq_folder || isempty(params.EMU_num)  
    if ~params.acq_is_beh && ~test_remote_folder(params.acq_remote_folder_in_beh)
        custompath.rm()
        error('remote acq folder not detected and needed to find EMU_num or to move files inside folder at the end')
    end
end

if isempty(params.EMU_num)
    allemus = dir(fullfile(params.acq_remote_folder_in_beh,['EMU-*' params.sub_ID '*']));
    allemus = {allemus.name};
    if exist(fullfile(params.acq_remote_folder_in_beh,'transferred'),'dir')
        transfemus = dir(fullfile(params.acq_remote_folder_in_beh,'transferred'));
        allemus = [allemus {transfemus.name}];
    end
    emulog_file = fullfile(params.acq_remote_folder_in_beh,['emulog_' params.sub_ID '.txt']); 
    if exist(emulog_file,'file')
        lines = readlines(emulog_file,'EmptyLineRule','skip');
        allemus = [allemus lines'];
    end
    if isempty(allemus)
        params.EMU_num = 1;
    else
        nums = zeros(1,length(allemus));
        for i = 1:length(allemus)
            num = regexp(allemus{i},['EMU-(\d+)_subj-', params.sub_ID, '*'],'tokens','once');
            if ~isempty(num)
                nums(i) = str2num(num{1});
            end
        end
        params.EMU_num = max(nums) + 1;        
    end
end

if params.with_acq_folder
    folder_name = sprintf('EMU-%.3d_subj-%s_task-gaps',params.EMU_num,params.sub_ID);
    folder_name = fullfile(params.acq_remote_folder_in_beh,folder_name);

    prev_exps = dir([folder_name '*']);
    if ~isempty(prev_exps)
        error('Filename or folder already exists. Please check everything and run again')
    end
end
    

% run_num=1;
run_num=0;
recording = recording_handler(params, '');


fig  = figure('Name', 'Recording','Toolbar','none','MenuBar','none','position',[300 400 340 50]);
set(fig,'CloseRequestFcn','')
uicontrol('Parent',fig, 'style','push', 'units','pix',...
                 'position',[10 5 130 30], 'fontsize',11,...
                 'string','Stop Recording', 'callback',{@(x,y) delete(gcf)});
uicontrol('Parent',fig, 'style','text', 'units','pix',...
                 'position',[150 5 150 30],...
                 'fontsize',9,'string','Pressing the button will stop the gaps task');

ME = [];
n_attempts=0;
logfile = struct;

loop_rec(fig,recording,params,folder_name,run_num,logfile,ME,n_attempts,rec_length)
custompath.rm()
% save_log(params,folder_name,logfile)
disp('GAPS completed succesfully')
end

function loop_rec(fig,recording,params,folder_name,run_num,logfile,ME,n_attempts,rec_length)
try    
    MEnew = ME;
    try
        recording.stop_with_connection_check()
    catch
        disp('recording not stopped (or already stopped)')
    end
    run_num = run_num + 1;
    started = false;
    abort = false;
    while ~abort
        if ~isgraphics(fig)
            abort = true;  
            break;
        end
        recording.rec_name = sprintf('EMU-%.3d_subj-%s_task-gaps_run-%.2d_%s',params.EMU_num,params.sub_ID,run_num,params.system);
        pause(0.3)
        recording.start()
        t0 = tic;
        logfile(run_num).file = recording.rec_name;
        logfile(run_num).start = datetime(now,'ConvertFrom','datenum'); 
        logfile(run_num).ME = ME; 
        fprintf('\nfile: %s, date: %s\n',logfile(run_num).file,logfile(run_num).start)
        started = true;
        prev_time = toc(t0);
        fprintf('status ')
        while (toc(t0)) < rec_length
            pause(1)
            if ~isgraphics(fig)
                abort = true;  
                break;
            end
            if toc(t0)-prev_time>floor(rec_length/6)
                recording.check_status
                fprintf('%s, ',recording.status)
                if ~strcmp(recording.status,'running') && ~strcmp(recording.status,'recording')
                    pause(5)
                    if ~strcmp(recording.status,'running') && ~strcmp(recording.status,'recording')
                        error('status not running')
                    end
                end
                prev_time = toc(t0);
            end
        end
        recording.stop_with_connection_check()
        run_num = run_num + 1;
        started = false;
    end 
    recording.close()
    
    run_num = run_num + 1;
    logfile(run_num).file = 'END OF SESSION';
    logfile(run_num).start = datetime(now,'ConvertFrom','datenum');    
    logfile(run_num).ME = ME; 
    save_log(params,folder_name,logfile,n_attempts,MEnew)  
catch MEnew
    logfile(run_num).ME = MEnew; 
    n_attempts = n_attempts + 1
    pause(5)     
    if n_attempts == 10
        main_text = sprintf('message: %s; file: %s; line: %d \n',MEnew.message,MEnew.stack(1).file,MEnew.stack(1).line);  
        alert_subject = 'gaps error';
        save(fullfile(params.acq_remote_folder_in_beh,sprintf('gaps_error_EMU%d.mat',params.EMU_num)))      
        save_log(params,folder_name,logfile,n_attempts,MEnew)  
        save(fullfile(folder_name,sprintf('gaps_error_EMU%d.mat',params.EMU_num)))      
        delete(fullfile(params.acq_remote_folder_in_beh,sprintf('gaps_error_EMU%d.mat',params.EMU_num)))      
%         send_email_toHGR(recording.rec_name,alert_subject,main_text)
        system("python3 gaps_error_email.py")        
        rethrow(MEnew)
    end
    loop_rec(fig,recording,params,folder_name,run_num,logfile,[],n_attempts,rec_length)
end
% custompath.rm()
% error('DONT PANIC !! FINISHED OK. just killing all recursive if necessary')
end


function save_log(params,folder_name,logfile,n_attempts,MEnew)

if params.with_acq_folder
    files = sprintf('EMU-%.3d_subj-%s_task-gaps_run-*',params.EMU_num,params.sub_ID);
    mkdir(folder_name)
    pause(5)
    movefile(fullfile(params.acq_remote_folder_in_beh,files),fullfile(folder_name));
    save(fullfile(folder_name,sprintf('EMU-%.3d_subj-%s_task-gaps_logfile.mat',params.EMU_num,params.sub_ID)),'logfile','n_attempts','MEnew')
end
end


% try
%     loop_rec(fig,recording,params,run_num,logfile,ME)
%     
%     abort = false;
%     while ~abort
%         recording.rec_name = sprintf('EMU-%.3d_subj-%s_task-gaps_run-%.2d_%s',params.EMU_num,params.sub_ID,run_num,params.system)
%         pause(0.3)
%         recording.start()
%         t0 = tic;
%         logfile(run_num).file = recording.rec_name;
%         logfile(run_num).start = datetime(now,'ConvertFrom','datenum');    
%         while (toc(t0)) < rec_length
%             if ~isgraphics(fig)
%                 abort = true;  
%                 break;
%             end
%             pause(0.3)
%         end
%         recording.stop_with_connection_check()
%         run_num = run_num + 1;
%     end 
%     recording.close()
%     
%     run_num = run_num + 1;
%     logfile(run_num).file = 'END OF SESSION';
%     logfile(run_num).start = datetime(now,'ConvertFrom','datenum');    
%     logfile(run_num).ME = []; 
% 
% 
% catch ME1
%     pause(5)
%     try
%         try
%             recording.stop_with_connection_check()
%         catch
%             disp('recording not stopped (or already stopped)')
%         end
%         run_num = run_num + 1;
%         while ~abort
%             recording.rec_name = sprintf('EMU-%.3d_subj-%s_task-gaps_run-%.2d_%s',params.EMU_num,params.sub_ID,run_num,params.system)
%             pause(0.3)
%             recording.start()
%             t0 = tic;
%             logfile(run_num).file = recording.rec_name;
%             logfile(run_num).start = datetime(now,'ConvertFrom','datenum');    
%             while (toc(t0)) < rec_length
%                 if ~isgraphics(fig)
%                     abort = true;  
%                     break;
%                 end
%                 pause(0.3)
%             end
%             recording.stop_with_connection_check()
%             run_num = run_num + 1;
%         end 
%         recording.close()
%         
%         run_num = run_num + 1;
%         logfile(run_num).file = 'END OF SESSION';
%         logfile(run_num).start = datetime(now,'ConvertFrom','datenum'); 
%         logfile(run_num).ME = ME1;    
%     catch ME2
%         main_text = sprintf('message1: %s; file1: %s; line1: %d \n message2: %s; file2: %s; line2: %d \n',ME1.message,ME1.stack(1).file,ME1.stack(1).line,ME2.message,ME2.stack(1).file,ME2.stack(1).line);  
%         alert_subject = 'gaps error';
%         save(fullfile(params.acq_remote_folder_in_beh,sprintf('gaps_error_EMU%s.mat',params.EMU_num)))      
%         save_log(params,folder_name,logfile)  
%         send_email_toHGR(recording.rec_name,alert_subject,main_text)
%         rethrow(ME2)
%     end
% end
% custompath.rm()
% save_log(params,folder_name)
% end