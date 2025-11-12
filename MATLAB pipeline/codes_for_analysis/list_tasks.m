function list_tasks(varargin)

cd('/mnt/data0/sEEG_DATA')
% [~,name] = system('hostname');
% current_user = getenv('USER');

[~,name] = system('hostname');
if contains(name,'BEH-REYLAB'), dir_base = '/home/user/share/codes_emu';
elseif contains(name,'TOWER-REYLAB') || contains(name,'RACK-REYLAB')
    %     current_user = 'sofiad';  % replace with appropriate user name
    current_user = getenv('USER');    dir_base = sprintf('/home/%s/Documents/GitHub/codes_emu',current_user);
elseif contains(name,'NSRG-HUB-15446'), dir_base = 'D:\codes_emu'; % Hernan's desktop
elseif contains(name,'NSRG-HUB-16167'), dir_base = 'D:\bcm_emu'; % Hernan's laptop
    %elseif contains(name,'ABT-REYLAB'), dir_base = 'C:\Users\user\Documents\GitHub\codes_emu'; % ABT-REYLAB
elseif contains(name,'ABT-REYLAB'), dir_base = 'C:\Users\smathew\OneDrive - mcw.edu\Rey lab\codes_emu'; % ABT-REYLAB
elseif contains(name,'MCW-20880'), dir_base = 'C:\Users\de31182\Documents\GitHub\codes_emu'; %Dewan's laptop
end

addpath(dir_base);

dir_b = sprintf('/home/%s/Documents/GitHub/codes_emu/codes_for_analysis',current_user);
addpath(dir_b);
custompath = reylab_custompath({'tasks/locations/'});
custompath = reylab_custompath({'wave_clus_reylab','NPMK-master_Gemini','codes_for_analysis','mex','useful_functions','neuroshare','tasks/.','tasks/locations/'});
dirmain = pwd;
listing = dir;
patients_log = [];

if isfile('Tasks_list.xlsx')
    DynamicScr_list = table2cell(readtable('Tasks_list.xlsx','Sheet','DynamicScr'));
    Recall_list = table2cell(readtable('Tasks_list.xlsx','Sheet','Recall'));
    MiniScr_list = table2cell(readtable('Tasks_list.xlsx','Sheet','OnlineMiniScr'));
    PictureNaming_list = table2cell(readtable('Tasks_list.xlsx','Sheet','PictureNaming'));
    DefinitionNaming_list = table2cell(readtable('Tasks_list.xlsx','Sheet','DefinitionNaming'));
    MusicAttention_list = table2cell(readtable('Tasks_list.xlsx','Sheet','MusicAttention'));
    MusicPassive_list = table2cell(readtable('Tasks_list.xlsx','Sheet','MusicPassive'));
    Movement_list = table2cell(readtable('Tasks_list.xlsx','Sheet','Movement'));
    DynamicSeman_list = table2cell(readtable('Tasks_list.xlsx','Sheet','DynamicSeman'));
    MusicFamiliar_list = table2cell(readtable('Tasks_list.xlsx','Sheet','MusicFamiliar'));
    MusicUnfamiliar_list = table2cell(readtable('Tasks_list.xlsx','Sheet','MusicUnfamiliar'));
    RSVPCategLocaliz_list = table2cell(readtable('Tasks_list.xlsx','Sheet','RSVPCategLocaliz'));
    PTSD_list = table2cell(readtable('Tasks_list.xlsx','Sheet','PTSD'));
    AudioBook_list = table2cell(readtable('Tasks_list.xlsx','Sheet','AudioBook'));
    SA_VS_OBA_list = table2cell(readtable('Tasks_list.xlsx','Sheet','SA_VS_OBA'));
    WheelOfFortune_list = table2cell(readtable('Tasks_list.xlsx','Sheet','WheelOfFortune'));
    RecDec_list = table2cell(readtable('Tasks_list.xlsx','Sheet','RecDec'));
    Conversation_list = table2cell(readtable('Tasks_list.xlsx','Sheet','Conversation'));
    Podcast_list = table2cell(readtable('Tasks_list.xlsx','Sheet','Podcast'));
else
    DynamicScr_list = [];
    Recall_list = [];
    MiniScr_list = [];
    PictureNaming_list = [];
    DefinitionNaming_list = [];
    MusicAttention_list = [];
    MusicPassive_list = [];
    Movement_list = [];
    DynamicSeman_list = [];
    MusicFamiliar_list = [];
    MusicUnfamiliar_list = [];
    RSVPCategLocaliz_list = [];
    PTSD_list = [];
    AudioBook_list = [];
    SA_VS_OBA_list = [];
    WheelOfFortune_list = [];
    RecDec_list = [];
    Conversation_list = [];
    Podcast_list = [];
end

for i = 1:length(listing)
    if contains(listing(i).name,'MCW-FH_')
        patients_log = [patients_log;{listing(i).name}];
    end
end

if isfile('subjects_log.txt')
    subjects_log = readlines('subjects_log.txt');
    subjects_log = cellstr(subjects_log);
    subjects_log(length(subjects_log)) = [];
    for m = 1:length(subjects_log)
        if length(patients_log)==1
            if strcmp(patients_log(1),subjects_log(m))
                patients_log(1) = [];
            end
        else
            for n = 1:length(patients_log)-1
                if strcmp(patients_log(n),subjects_log(m))
                    patients_log(n) = [];
                end
            end
        end
    end
end

for j=1:length(patients_log)
    current_dir = char(fullfile(dirmain,'/',patients_log(j),'/EMU/'));
    cd(current_dir)
    Tasks_list = dir;
    for k = 1:length(Tasks_list)
        if contains(Tasks_list(k).name,'RSVPscr') || contains(Tasks_list(k).name,'RSVPDynamicScr') || contains(Tasks_list(k).name,'RSVPdynamic')
            c_dir = char(fullfile(current_dir,'/',Tasks_list(k).name));
            cd(c_dir);
            d_time = string(sprintf('%s_RIP.ns5',Tasks_list(k).name));
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
            Info_task = [{patients_log(j)},{Tasks_list(k).name},{Tasks_list(k).folder},Start_Time,End_Time, dt];
            DynamicScr_list = [DynamicScr_list;Info_task];
        elseif contains(Tasks_list(k).name,'Recall') || contains(Tasks_list(k).name,'recall')
            c_dir = char(fullfile(current_dir,'/',Tasks_list(k).name));
            cd(c_dir);
            d_time = string(sprintf('%s_RIP.ns5',Tasks_list(k).name));
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
            Info_task = [{patients_log(j)},{Tasks_list(k).name},{Tasks_list(k).folder},Start_Time,End_Time, dt];
            Recall_list = [Recall_list;Info_task];
        elseif contains(Tasks_list(k).name,'RSVPmini') || contains(Tasks_list(k).name,'RSVPOnlineMiniScr')
            c_dir = char(fullfile(current_dir,'/',Tasks_list(k).name));
            cd(c_dir);
            d_time = string(sprintf('%s_RIP.ns5',Tasks_list(k).name));
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
            Info_task = [{patients_log(j)},{Tasks_list(k).name},{Tasks_list(k).folder},Start_Time,End_Time, dt];
            MiniScr_list = [MiniScr_list;Info_task];
        elseif contains(Tasks_list(k).name,'PictureNaming')
            c_dir = char(fullfile(current_dir,'/',Tasks_list(k).name));
            cd(c_dir);
            d_time = string(sprintf('%s_RIP.ns5',Tasks_list(k).name));
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
            Info_task = [{patients_log(j)},{Tasks_list(k).name},{Tasks_list(k).folder},Start_Time,End_Time, dt];
            PictureNaming_list = [PictureNaming_list;Info_task];
        elseif contains(Tasks_list(k).name,'DefinitionNaming')
            c_dir = char(fullfile(current_dir,'/',Tasks_list(k).name));
            cd(c_dir);
            d_time = string(sprintf('%s_RIP.ns5',Tasks_list(k).name));
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
            Info_task = [{patients_log(j)},{Tasks_list(k).name},{Tasks_list(k).folder},Start_Time,End_Time, dt];
            DefinitionNaming_list = [DefinitionNaming_list;Info_task];
        elseif contains(Tasks_list(k).name,'MusicAttention')
            c_dir = char(fullfile(current_dir,'/',Tasks_list(k).name));
            cd(c_dir);
            d_time = string(sprintf('%s_RIP.ns5',Tasks_list(k).name));
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
            Info_task = [{patients_log(j)},{Tasks_list(k).name},{Tasks_list(k).folder},Start_Time,End_Time, dt];
            MusicAttention_list = [MusicAttention_list;Info_task];
        elseif contains(Tasks_list(k).name,'MusicPassive')
            c_dir = char(fullfile(current_dir,'/',Tasks_list(k).name));
            cd(c_dir);
            d_time = string(sprintf('%s_RIP.ns5',Tasks_list(k).name));
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
            Info_task = [{patients_log(j)},{Tasks_list(k).name},{Tasks_list(k).folder},Start_Time,End_Time, dt];
            MusicPassive_list = [MusicPassive_list;Info_task];
        elseif contains(Tasks_list(k).name,'movement') || contains(Tasks_list(k).name,'Movement')
            c_dir = char(fullfile(current_dir,'/',Tasks_list(k).name));
            cd(c_dir);
            d_time = string(sprintf('%s_RIP.ns5',Tasks_list(k).name));
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
            Info_task = [{patients_log(j)},{Tasks_list(k).name},{Tasks_list(k).folder},Start_Time,End_Time, dt];
            Movement_list = [Movement_list;Info_task];
        elseif contains(Tasks_list(k).name,'RSVPDynamicSeman')
            c_dir = char(fullfile(current_dir,'/',Tasks_list(k).name));
            cd(c_dir);
            d_time = string(sprintf('%s_RIP.ns5',Tasks_list(k).name));
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
            Info_task = [{patients_log(j)},{Tasks_list(k).name},{Tasks_list(k).folder},Start_Time,End_Time, dt];
            DynamicSeman_list = [DynamicSeman_list;Info_task];
        elseif contains(Tasks_list(k).name,'MusicFamiliar')
            c_dir = char(fullfile(current_dir,'/',Tasks_list(k).name));
            cd(c_dir);
            d_time = string(sprintf('%s_RIP.ns5',Tasks_list(k).name));
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
            Info_task = [{patients_log(j)},{Tasks_list(k).name},{Tasks_list(k).folder},Start_Time,End_Time, dt];
            MusicFamiliar_list = [MusicFamiliar_list;Info_task];
        elseif contains(Tasks_list(k).name,'MusicUnfamiliar')
            c_dir = char(fullfile(current_dir,'/',Tasks_list(k).name));
            cd(c_dir);
            d_time = string(sprintf('%s_RIP.ns5',Tasks_list(k).name));
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
            Info_task = [{patients_log(j)},{Tasks_list(k).name},{Tasks_list(k).folder},Start_Time,End_Time, dt];
            MusicUnfamiliar_list = [MusicUnfamiliar_list;Info_task];
        elseif contains(Tasks_list(k).name,'RSVPCategLocaliz')
            c_dir = char(fullfile(current_dir,'/',Tasks_list(k).name));
            cd(c_dir);
            d_time = string(sprintf('%s_RIP.ns5',Tasks_list(k).name));
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
            Info_task = [{patients_log(j)},{Tasks_list(k).name},{Tasks_list(k).folder},Start_Time,End_Time, dt];
            RSVPCategLocaliz_list = [RSVPCategLocaliz_list;Info_task];
        elseif contains(Tasks_list(k).name,'PTSD')
            c_dir = char(fullfile(current_dir,'/',Tasks_list(k).name));
            cd(c_dir);
            d_time = string(sprintf('%s_RIP.ns5',Tasks_list(k).name));
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
            Info_task = [{patients_log(j)},{Tasks_list(k).name},{Tasks_list(k).folder},Start_Time,End_Time, dt];
            PTSD_list = [PTSD_list;Info_task];
        elseif contains(Tasks_list(k).name,'AudioBook')
            c_dir = char(fullfile(current_dir,'/',Tasks_list(k).name));
            cd(c_dir);
            d_time = string(sprintf('%s_RIP.ns5',Tasks_list(k).name));
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
            Info_task = [{patients_log(j)},{Tasks_list(k).name},{Tasks_list(k).folder},Start_Time,End_Time, dt];
            AudioBook_list = [AudioBook_list;Info_task];
        elseif contains(Tasks_list(k).name,'SA_VS_OBA')
            c_dir = char(fullfile(current_dir,'/',Tasks_list(k).name));
            cd(c_dir);
            d_time = string(sprintf('%s.ns5',Tasks_list(k).name));
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
            Info_task = [{patients_log(j)},{Tasks_list(k).name},{Tasks_list(k).folder},Start_Time,End_Time, dt];
            SA_VS_OBA_list = [SA_VS_OBA_list;Info_task];
        elseif contains(Tasks_list(k).name,'WheelOfFortune')
            c_dir = char(fullfile(current_dir,'/',Tasks_list(k).name));
            cd(c_dir);
            d_time = string(sprintf('%s_RIP.ns5',Tasks_list(k).name));
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
            Info_task = [{patients_log(j)},{Tasks_list(k).name},{Tasks_list(k).folder},Start_Time,End_Time, dt];
            WheelOfFortune_list = [WheelOfFortune_list;Info_task];
        elseif contains(Tasks_list(k).name,'RecDec')
            c_dir = char(fullfile(current_dir,'/',Tasks_list(k).name));
            cd(c_dir);
            d_time = string(sprintf('%s.ns5',Tasks_list(k).name));
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
            Info_task = [{patients_log(j)},{Tasks_list(k).name},{Tasks_list(k).folder},Start_Time,End_Time, dt];
            RecDec_list = [RecDec_list;Info_task];
        elseif contains(Tasks_list(k).name,'Conversation')
            c_dir = char(fullfile(current_dir,'/',Tasks_list(k).name));
            cd(c_dir);
            d_time = string(sprintf('%s_RIP.ns5',Tasks_list(k).name));
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
            Info_task = [{patients_log(j)},{Tasks_list(k).name},{Tasks_list(k).folder},Start_Time,End_Time, dt];
            Conversation_list = [Conversation_list;Info_task];
        elseif contains(Tasks_list(k).name,'Podcast')
            c_dir = char(fullfile(current_dir,'/',Tasks_list(k).name));
            cd(c_dir);
            d_time = string(sprintf('%s_RIP.ns5',Tasks_list(k).name));
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
            Info_task = [{patients_log(j)},{Tasks_list(k).name},{Tasks_list(k).folder},Start_Time,End_Time, dt];
            Podcast_list = [Podcast_list;Info_task];
        end
    end
    cd(dirmain)
end

filename = 'Tasks_list.xlsx';
columnNames = {'Patient','File name','Path','Start time', 'End time', 'Duration'};
writetable(array2table(DynamicScr_list,"VariableNames",columnNames),filename,'Sheet','DynamicScr');
writetable(array2table(Recall_list,"VariableNames",columnNames),filename,'Sheet','Recall');
writetable(array2table(MiniScr_list,"VariableNames",columnNames),filename,'Sheet','OnlineMiniScr');
writetable(array2table(PictureNaming_list,"VariableNames",columnNames),filename,'Sheet','PictureNaming');
writetable(array2table(DefinitionNaming_list,"VariableNames",columnNames),filename,'Sheet','DefinitionNaming');
writetable(array2table(MusicAttention_list,"VariableNames",columnNames),filename,'Sheet','MusicAttention');
writetable(array2table(MusicPassive_list,"VariableNames",columnNames),filename,'Sheet','MusicPassive');
writetable(array2table(Movement_list,"VariableNames",columnNames),filename,'Sheet','Movement');
writetable(array2table(RSVPCategLocaliz_list,"VariableNames",columnNames),filename,'Sheet','RSVPCategLocaliz');
writetable(array2table(DynamicSeman_list,"VariableNames",columnNames),filename,'Sheet','DynamicSeman');
writetable(array2table(MusicFamiliar_list,"VariableNames",columnNames),filename,'Sheet','MusicFamiliar');
writetable(array2table(AudioBook_list,"VariableNames",columnNames),filename,'Sheet','AudioBook');
writetable(array2table(MusicUnfamiliar_list,"VariableNames",columnNames),filename,'Sheet','MusicUnfamiliar');
writetable(array2table(PTSD_list,"VariableNames",columnNames),filename,'Sheet','PTSD');
writetable(array2table(SA_VS_OBA_list,"VariableNames",columnNames),filename,'Sheet','SA_VS_OBA');
writetable(array2table(WheelOfFortune_list,"VariableNames",columnNames),filename,'Sheet','WheelOfFortune');
writetable(array2table(RecDec_list,"VariableNames",columnNames),filename,'Sheet','RecDec');
writetable(array2table(Conversation_list,"VariableNames",columnNames),filename,'Sheet','Conversation');
writetable(array2table(Podcast_list,"VariableNames",columnNames),filename,'Sheet','Podcast');

if isfile('subjects_log.txt')
    subjects_log = cellstr(subjects_log);
    patients_log = [subjects_log;patients_log];
    writecell(patients_log,'subjects_log.txt');
else
    writecell(patients_log,'subjects_log.txt');
end

end
