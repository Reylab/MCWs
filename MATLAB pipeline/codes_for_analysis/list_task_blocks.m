function list_task_blocks(varargin)

cd('/mnt/data0/sEEG_DATA')
% cd('/media/sEEG_DATA/')
[~,name] = system('hostname');
current_user = getenv('USER');
dir_base = sprintf('/home/%s/Documents/GitHub/codes_emu/codes_for_analysis',current_user);
addpath(dir_base);
custompath = reylab_custompath({'tasks/locations/'});
custompath = reylab_custompath({'wave_clus_reylab','NPMK-master_Gemini','codes_for_analysis','mex','useful_functions','neuroshare','tasks/.','tasks/locations/'});
dirmain = pwd;
listing = dir;
patients_log = [];

if isfile('Task_blocks.xlsx')
    DynamicScr_list = table2cell(readtable('Task_blocks.xlsx','Sheet','DynamicScr'));
    MusicFamiliar_list = table2cell(readtable('Task_blocks.xlsx','Sheet','MusicFamiliar'));
else
    DynamicScr_list = [];
    MusicFamiliar_list = [];
end

Dynamic_list = [];
MusicF_list = [];

for i = 1:length(listing)
    if contains(listing(i).name,'MCW-FH_')
        patients_log = [patients_log;{listing(i).name}];
    end
end

if isfile('subjects_block_log.txt')
    subjects_block_log = readlines('subjects_block_log.txt');
    subjects_block_log = cellstr(subjects_block_log);
    subjects_block_log(length(subjects_block_log)) = [];
    for m = 1:length(subjects_block_log)
        if length(patients_log)==1
            if strcmp(patients_log(1),subjects_block_log(m))
                patients_log(1) = [];
            end
        else
            for n = 1:length(patients_log)-1
                if strcmp(patients_log(n),subjects_block_log(m))
                    patients_log(n) = [];
                end
            end
        end
    end
end

for j=1:length(patients_log)
    current_dir = char(fullfile(dirmain,'/',patients_log(j),'/EMU/'));
    cd(current_dir)
    Task_blocks = dir;
    for k = 1:length(Task_blocks)
        if (contains(Task_blocks(k).name,'RSVPscr') || contains(Task_blocks(k).name,'RSVPDynamicScr') || contains(Task_blocks(k).name,'RSVPdynamic')) && (~contains(Task_blocks(k).name,'error') && ~contains(Task_blocks(k).name,'seizure') && ~contains(Task_blocks(k).name,'BAD'))
            f_dynamic = false;
            c_dir = char(fullfile(current_dir,'/',Task_blocks(k).name));
            cd(c_dir);
            d_time = string(sprintf('%s_RIP.ns5',Task_blocks(k).name));
            for i =1:4
                if length(Task_blocks)>=k+i
                    if contains(Task_blocks(k+i).name,'RSVPscr') || contains(Task_blocks(k+i).name,'RSVPDynamicScr')
                        m = i;
                    end
                end
            end
            if ~exist('m','var')
                m = 4;
            end
            for ii = 1:m
                if length(Task_blocks)>=k+ii
                    if contains(Task_blocks(k+ii).name,'Recall') || contains(Task_blocks(k+ii).name,'recall') || contains(Task_blocks(k+ii).name,'RSVPmini') || contains(Task_blocks(k+ii).name,'RSVPOnlineMiniScr')
                        if f_dynamic == false
                            if exist(d_time,'file') == 2
                                [Start_Time,End_Time] = get_rip_starttime(d_time);
                                dt = End_Time - Start_Time;
                                Start_Time = string(Start_Time);
                                End_Time = string(End_Time);
                                dt = string(dt);
                            else
                                Start_Time = string('error');
                                End_Time = string('error');
                                dt = string('error');
                            end
                            Info_task = [{patients_log(j)},{Task_blocks(k).name},{Task_blocks(k).folder},Start_Time,End_Time, dt];
                            Dynamic_list = [Dynamic_list;Info_task];
                            f_dynamic = true;
                        end
                        cd(current_dir)

                        c_dir = char(fullfile(current_dir,'/',Task_blocks(k+ii).name));
                        cd(c_dir)
                        d_time = string(sprintf('%s_RIP.ns5',Task_blocks(k+ii).name));
                        if exist(d_time,'file') == 2
                            [Start_Time,End_Time] = get_rip_starttime(d_time);
                            dt = End_Time - Start_Time;
                            Start_Time = string(Start_Time);
                            End_Time = string(End_Time);
                            dt = string(dt);
                        else
                            Start_Time = string('error');
                            End_Time = string('error');
                            dt = string('error');
                        end
                        Info_task = [{patients_log(j)},{Task_blocks(k+ii).name},{Task_blocks(k+ii).folder},Start_Time,End_Time, dt];
                        Dynamic_list = [Dynamic_list;Info_task];
                        cd(current_dir)
                        %                     elseif contains(Task_blocks(k+i).name,'RSVPmini') || contains(Task_blocks(k+i).name,'RSVPOnlineMiniScr')
                        %                         c_dir = char(fullfile(current_dir,'/',Task_blocks(k+i).name));
                        %                         cd(c_dir)
                        %                         d_time = string(sprintf('%s_RIP.ns5',Task_blocks(k+i).name));
                        %                         if exist(d_time,'file') == 2
                        %                             [Start_Time,End_Time] = get_rip_starttime(d_time);
                        %                             Start_Time = string(Start_Time);
                        %                             End_Time = string(End_Time);
                        %                         else
                        %                             Start_Time = string('error');
                        %                             End_Time = string('error');
                        %                         end
                        %                         Info_task = [{patients_log(j)},{Task_blocks(k+i).name},{Task_blocks(k+i).folder},Start_Time,End_Time];
                        %                         Dynamic_list = [Dynamic_list;Info_task];
                        %                         cd(current_dir)
                    end
                end
            end

            %                 if ~isempty(Dynamic_list)
            %                     s = size(Dynamic_list);
            %                     a = Dynamic_list{s(1),:};
            %                     if length(a)>1
            %                         Info_task = [{' '},{' '},{' '},{' '},{' '}];
            %                         Dynamic_list = [Dynamic_list;Info_task];
            %                     end
            %                 end
            if ~isempty (Dynamic_list)
                for i = 1:size(Dynamic_list, 1)
                    if i == 1
                        new_info_task = Dynamic_list(i,:);
                        DynamicScr_list = [DynamicScr_list;new_info_task];
                    else
                        Start_Time = Dynamic_list(i-1,5);
                        End_Time = Dynamic_list(i,4);
                        dt = datetime(End_Time) - datetime(Start_Time);
                        dt = string(dt);

                        gap = ['-','Inter task interval','-','-','-', dt];
                        DynamicScr_list = [DynamicScr_list;gap];
                        new_info_task = Dynamic_list(i,:);
                        DynamicScr_list = [DynamicScr_list;new_info_task];

                    end
                end
                Dynamic_list = [];
            end

            if ~isempty(DynamicScr_list)
                s = size(DynamicScr_list);
                a = DynamicScr_list{s(1),:};
                if length(a)>1
                    Info_task = [{' '},{' '},{' '},{' '},{' '},{' '}];
                    DynamicScr_list = [DynamicScr_list;Info_task;Info_task];
                end
            end

        elseif contains(Task_blocks(k).name,'MusicFamiliar')
            for i =1:4

                if find(strcmp({Task_blocks.name},Task_blocks(16).name))>4
                    if contains(Task_blocks(k-i).name,'OnlineMiniScrMusic')
                        c_dir = char(fullfile(current_dir,'/',Task_blocks(k-i).name));
                        cd(c_dir)
                        d_time = string(sprintf('%s_RIP.ns5',Task_blocks(k-i).name));
                        if exist(d_time,'file') == 2
                            [Start_Time,End_Time] = get_rip_starttime(d_time);
                            dt = End_Time - Start_Time;
                            Start_Time = string(Start_Time);
                            End_Time = string(End_Time);
                            dt = string(dt);
                        else
                            Start_Time = string('error');
                            End_Time = string('error');
                            dt = string('error');
                        end
                        Info_task = [{patients_log(j)},{Task_blocks(k-i).name},{Task_blocks(k-i).folder},Start_Time,End_Time, dt];
                        MusicF_list = [MusicF_list;Info_task];
                        cd(current_dir)

                        c_dir = char(fullfile(current_dir,'/',Task_blocks(k).name));
                        cd(c_dir)
                        d_time = string(sprintf('%s_RIP.ns5',Task_blocks(k).name));
                        if exist(d_time,'file') == 2
                            [Start_Time,End_Time] = get_rip_starttime(d_time);
                            dt = End_Time - Start_Time;
                            Start_Time = string(Start_Time);
                            End_Time = string(End_Time);
                            dt = string(dt);
                        else
                            Start_Time = string('error');
                            End_Time = string('error');
                            dt = string('error');
                        end
                        Info_task = [{patients_log(j)},{Task_blocks(k).name},{Task_blocks(k).folder},Start_Time,End_Time, dt];
                        MusicF_list = [MusicF_list;Info_task];
                        cd(current_dir)
                    end
                end
                if length(Task_blocks)>=k+i
                    if contains(Task_blocks(k+i).name,'OnlineMiniScrMusic')
                        c_dir = char(fullfile(current_dir,'/',Task_blocks(k+i).name));
                        cd(c_dir)
                        d_time = string(sprintf('%s_RIP.ns5',Task_blocks(k+i).name));
                        if exist(d_time,'file') == 2
                            [Start_Time,End_Time] = get_rip_starttime(d_time);
                            dt = End_Time - Start_Time;
                            Start_Time = string(Start_Time);
                            End_Time = string(End_Time);
                            dt = string(dt);
                        else
                            Start_Time = string('error');
                            End_Time = string('error');
                            dt = string('error');
                        end
                        Info_task = [{patients_log(j)},{Task_blocks(k+i).name},{Task_blocks(k+i).folder},Start_Time,End_Time, dt];
                        MusicF_list = [MusicF_list;Info_task];
                        cd(current_dir)
                    end
                end
            end

            if ~isempty (MusicF_list)
                for i = 1:size(MusicF_list, 1)
                    if i == 1
                        new_info_task = MusicF_list(i,:);
                        MusicFamiliar_list = [MusicFamiliar_list;new_info_task];
                    else
                        Start_Time = MusicF_list(i-1,5);
                        End_Time = MusicF_list(i,4);
                        dt = datetime(End_Time) - datetime(Start_Time);
                        dt = string(dt);

                        gap = ['-','Inter task interval','-','-','-', dt];
                        MusicFamiliar_list = [MusicFamiliar_list;gap];
                        new_info_task = MusicF_list(i,:);
                        MusicFamiliar_list = [MusicFamiliar_list;new_info_task];

                    end
                end
                MusicF_list = [];
            end

            if ~isempty(MusicFamiliar_list)
                s = size(MusicFamiliar_list);
                a = MusicFamiliar_list{s(1),:};
                if length(a)>1
                    Info_task = [{' '},{' '},{' '},{' '},{' '},{' '}];
                    MusicFamiliar_list = [MusicFamiliar_list;Info_task;Info_task];
                end
            end

        end

    end
    cd(dirmain)
end

filename = 'Task_blocks.xlsx';
columnNames = {'Patient','File name','Path','Start time', 'End time','Duration'};
writetable(array2table(DynamicScr_list,"VariableNames",columnNames),filename,'Sheet','DynamicScr');
writetable(array2table(MusicFamiliar_list,"VariableNames",columnNames),filename,'Sheet','MusicFamiliar');

if isfile('subjects_block_log.txt')
    subjects_block_log = cellstr(subjects_block_log);
    patients_log = [subjects_block_log;patients_log];
    writecell(patients_log,'subjects_block_log.txt');
else
    writecell(patients_log,'subjects_block_log.txt');
end

end
