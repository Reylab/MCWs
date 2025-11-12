function [Start_Time,End_Time] = get_rip_starttime(filenames)

if ~exist('which_system_micro','var')|| isempty(which_system_micro),  which_system_micro = 'RIP'; end 

[~,name] = system('hostname');
current_user = getenv('USER');    dir_base = sprintf('/home/%s/Documents/GitHub/codes_emu',current_user); 
addpath(dir_base);
custompath = reylab_custompath({'tasks/locations/'});

% if contains(name,'REYLAB')
%     params = MCW_location(['MCW-' which_system_micro]);
%     if ~exist('hours_offset','var')|| isempty(hours_offset)
%         hours_offset = params.offset;
%     end
% end

if ~exist('filenames','var')
   aux=dir('*.ns5');
   filenames= {aux.name};
end

expr2remove = '-\d+$';
%%
if ~isempty(mfilename)
    root_rc = [fileparts(mfilename('fullpath')) filesep '..'];
    if exist([root_rc filesep 'reylab_custompath.m'],'file')
        addpath(root_rc);
        custom_path = reylab_custompath('neuroshare');
    end
end

if ischar(filenames)
    filenames = {filenames};
end

    tic
    [ns_status, hFile] = ns_OpenFile(filenames{1}, 'single');
%     [ns_status_nev, hFile_nev] = ns_OpenFile([filename(1:end-3) 'nev'], 'single');
%     [ns_RESULT, nsFileInfo] = ns_GetFileInfo(hFile_nev);
    
    fid = fopen(filenames{1}, 'rb');
%     fid = fopen([filenames{1}(1:end-3) 'nev'], 'rb');
%     fseek(fid, 28, -1);
    fseek(fid, 294, -1);
    Date = fread(fid, 8, 'uint16');
    tUTC = datetime([Date(1:2);Date(4:7)]','TimeZone','America/Chicago');
    [dt,dst] = tzoffset(tUTC);
    Start_Time = tUTC+dt;

%     Start_Time = sprintf('%d/%d/%d %d:%d:%d',Date(2),Date(4),Date(1),Date(5),Date(6),Date(7));
%     Start_Time = datetime([Date(1:2);Date(4:7)]')-hours(hours_offset); %this is asuming local time 5 hour behind summit time
%      Start_Time = datetime([Date(1:2);Date(4:7)]');

    Rec_length_sec = hFile.TimeSpan/30000;
%     End_Time = datestr(datenum(Start_Time) + Rec_length_sec/86400, 'mm/dd/yyyy HH:MM:SS');
    End_Time = Start_Time + seconds(Rec_length_sec);

    Date_Time = [Start_Time;End_Time];
    

