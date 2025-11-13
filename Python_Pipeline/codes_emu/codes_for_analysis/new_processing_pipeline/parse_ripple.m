function parse_ripple(filenames,remove_chs,macro,max_memo_GB, overwrite, channels,which_system_micro,hifreq_raw)

% This code requires the neuroshape library in the path.
% max_memo_GB is an idea of the number of GB allocated for the data to be
% stored in RAM, so it is used to compute the number of segments in which
% the data should be split for processing

if ~exist('which_system_micro','var')|| isempty(which_system_micro),  which_system_micro = 'RIP'; end 
if ~exist('hifreq_raw','var')|| isempty(hifreq_raw),  hifreq_raw = 'hifreq'; end

% [~,name] = system('hostname');
% % current_user = getenv('USER');    dir_base = sprintf('/home/%s/Documents/GitHub/codes_emu',current_user); 
% % addpath(dir_base);
% % custompath = reylab_custompath({'tasks/locations/'});

% if contains(name,'REYLAB')
%     params = MCW_location(['MCW-' which_system_micro]);
%     hours_offset = params.offset;
% end

if ~exist('hours_offset','var')|| isempty(hours_offset), hours_offset = 0; end % time difference with Coordinated Universal Time (UTC)

if ~exist('filenames','var')
   aux=dir('*.ns5');
   filenames= {aux.name};
end

expr2remove = '-\d+$';
%%
if ~isempty(mfilename)
    root_rc = [fileparts(mfilename('fullpath')) filesep '..' filesep '..'];
    if exist([root_rc filesep 'reylab_custompath.m'],'file')
        addpath(root_rc);
        custom_path = reylab_custompath('neuroshare');
    end
end

%%


if ~exist('overwrite','var') || isempty(overwrite)
    overwrite = false;
end

if ~exist('macro','var')
    macro = [];
end

if ~exist('remove_chs','var')
    remove_chs = [];
end

if ispc
    [~, systemview] = memory;
    memo_available = floor(systemview.PhysicalMemory.Available*0.80);
elseif ismac
        memo_available = 12*(1024)^3; % reduce if necessary
%     command='(top -l 1 | grep PhysMem: | awk ''{print $6}'')';
%     [status,cmdout] = system(command);
%     if status == 0
%         memo_available = floor(str2double(cmdout(1:end-3))*0.8)*(1024)^3;
%     else
%         memo_available = 12*(1024)^3; % reduce if necessary
%     end
else
    command='(free -h | awk ''/^Mem:/ {print $7}'')';
    [status,cmdout] = system(command);
    if status == 0
        memo_available = floor(str2double(cmdout(1:end-3))*0.8)*(1024)^3;
    else
        memo_available = 12*(1024)^3; % reduce if necessary
    end
end

if exist('max_memo_GB','var') && ~isempty(max_memo_GB)
    max_memo = max_memo_GB*(1024)^3;
    if max_memo > memo_available
        error('max_memo_GB > 80% of Physical Memory Available')
    end
else
    max_memo = memo_available;
end

tcum=0;

if ischar(filenames)
    filenames = {filenames};
end

% formatvector=@(v,f) sprintf(['[' repmat(['%' f ', '], 1, numel(v)-1), ['%' f ']\n']  ],v);

metadata_file = fullfile(pwd, 'NSx.mat');
if exist(metadata_file,'file')
    metadata = load(metadata_file);
else
    metadata = [];
end

for fi = 1:length(filenames)
    filename= filenames{fi};
    new_files(fi).name = filename;
    if length(filename)<3 || (~strcmpi(filename(2:3),':\') && ...
                     ~strcmpi(filename(1),'/') && ...
                     ~strcmpi(filename(2),'/') && ...
                     ~strcmpi(filename(1:2), '\\')&& ~strcmpi(filename(2:3),':/'))
        filename= [pwd filesep filename];
    end
    
    tic
     [ns_status, hFile] = ns_OpenFile(filenames{1}, 'single');  
%     [ns_status, hFile] = ns_OpenFile(filename, 'single');
%     [ns_status_nev, hFile_nev] = ns_OpenFile([filename(1:end-3) 'nev'], 'single');
%     [ns_RESULT, nsFileInfo] = ns_GetFileInfo(hFile_nev);
    
    fid = fopen(filenames{1}, 'rb');
%     fid = fopen([filename(1:end-3) 'nev'], 'rb');
    fseek(fid, 294, -1);
%     fseek(fid, 28, -1);
    Date = fread(fid, 8, 'uint16');
%     nsFileInfo.Time_Year = Date(1);
%     nsFileInfo.Time_Month = Date(2);
%     nsFileInfo.Time_Day = Date(4);
%     nsFileInfo.Time_Hour = Date(5);
%     nsFileInfo.Time_Min = Date(6);
%     nsFileInfo.Time_Sec = Date(7);
    
    tUTC = datetime([Date(1:2);Date(4:7)]','TimeZone','America/Chicago');
    [dt,dst] = tzoffset(tUTC);
    Start_Time = tUTC+dt;

%     Start_Time = datetime([Date(1:2);Date(4:7)]')-hours(hours_offset); 
%     Date_Time = sprintf('%d/%d/%d %d:%d:%d',Date(2),Date(4),Date(1),Date(5),Date(6),Date(7));
%     Date_Time = sprintf('%d/%d/%d %d:%d:%d',nsFileInfo.Time_Month,nsFileInfo.Time_Day,nsFileInfo.Time_Year,nsFileInfo.Time_Hour,nsFileInfo.Time_Min,nsFileInfo.Time_Sec);

    Rec_length_sec = hFile.TimeSpan/30000;
%     End_Time = datestr(datenum(Start_Time) + Rec_length_sec/86400, 'mm/dd/yyyy HH:MM:SS');
    End_Time = Start_Time + seconds(Rec_length_sec);

    Date_Time = [Start_Time;End_Time];
    
    if strcmp(ns_status,'ns_FILEERROR')
        error('Unable to open file: %s',filename)
    end
    
    sr=str2double(hFile.FileInfo.Label(1:strfind(hFile.FileInfo.Label,'ksamp')-2))*1000;
    electrodes_row = [];
    if sr == 7500
        for i = 1:length(hFile.Entity)
            if contains(hFile.Entity(i).Label,hifreq_raw)
                electrodes_row = horzcat(electrodes_row, i);
            end
        end
    else
        electrodes_row = 1:length(hFile.Entity);
    end
    %     switch hFile.FileInfo.Type(3)
%         case '1'
%             sr = 500;
%         case '2'
%             sr = 1000;
%         case '3'
%             sr = 2000;
%         case '4'
%             sr = 10000;            
%         case {'5' ,'6'}
%             sr = 30000;
%         otherwise
%             error('ERROR: %s file type not supported',hFile.FileInfo.Type)
%     end
    
    nchan = size(hFile.Entity,2);   % number of channels
    nchan = size(electrodes_row,2);
    samples_per_channel = ceil(max_memo/(nchan*2));
    if fi == 1
        %get info about channels in nsx file
        chs_info = struct();
        chs_info.unit = cellfun(@(x) deblank(x),{hFile.Entity(electrodes_row).Units},'UniformOutput',false);
        chs_info.label = cellfun(@(x) deblank(x),{hFile.Entity(electrodes_row).Label},'UniformOutput',false)';
        chs_info.conversion = cell2mat({hFile.Entity(electrodes_row).Scale});
        chs_info.id = cell2mat({hFile.Entity(electrodes_row).ElectrodeID});
        chs_info.pak_list = 0*chs_info.id;
        chs_info.dc= double(0*chs_info.id);
        chs_info.macro = chs_info.label;
        micros = cellfun(@(x) strcmp(x,'uV'),chs_info.unit);
        if ~isempty(macro)
            chs_info.macro(micros) = arrayfun(@(x) macro{ceil(x/9)},find(micros),'UniformOutput',false);
        end
        
        outfile_handles = cell(1,nchan); %some will be empty
        [~,~,fext] = fileparts(filename);
        fext = lower(fext(2:end));
        nsx_ext = fext(end);
        ch_ext = ['.NC' nsx_ext];
        if ~exist('channels','var') || isempty(channels)
            channels = hFile.FileInfo.ElectrodeList(electrodes_row);
        end

        remove_channels_by_label = {'(ref(.*))$'};
        for ci=1:numel(channels)
            if ~isempty(regexp(chs_info.label{ci},remove_channels_by_label{1},'match'))
                remove_chs = [remove_chs channels(ci)];
            end
        end
        remove_chs = unique(remove_chs);

        if ~isempty(remove_chs)
            channels = setdiff(channels, remove_chs);
        end
        parsed_chs = [];
        new_channel_id = [];
        k=0;
        for i = 1:length(hFile.Entity)
           if ismember(i,electrodes_row)
                k = k+1;
                c = hFile.FileInfo.ElectrodeList(i);
                if ismember(c,channels)
                    ccname = c;
                    if ~isempty(metadata) %NSx file in current folder
                        if sr == 7500
                            repetead = arrayfun(@(x) (x.chan_ID==ccname) && endsWith(x.label,hifreq_raw),metadata.NSx);
                        else 
                            repetead = arrayfun(@(x) (x.chan_ID==ccname) && (x.sr==sr),metadata.NSx);
                        end
                        if ~isempty(repetead) && sum(repetead)>0 %found channel
                            pos = find(repetead);
                            if overwrite
                                f2delete = [metadata.NSx(pos).output_name  metadata.NSx(pos).ext];
                                fprintf('Overwritting channel %d. Deleting file %s\n', metadata.NSx(pos).chan_ID, f2delete)
                                delete(f2delete)
                            else
                                fprintf('Skipping channel %d, already parsed.\n', metadata.NSx(pos).chan_ID)
                                continue %If output_name wasn't set, the existing parsed channels won't be overwritten.
                            end
                        end
                    end
                    parsed_chs(end+1) = c;
                    new_channel_id(end+1) = ccname;
                    
                    
                    ix = chs_info.id==c;
                    
                    %output_name = chs_info.label{ix};
                    output_name = chs_info.macro{ix};
                    outn_i = regexp(output_name,expr2remove);
                    if ~isempty(outn_i) && outn_i>1
                        output_name = output_name(1:outn_i-1);
                    end
                    %                 ix = find(ix,1);
                    chs_info.output_name{ix} = [output_name '_' num2str(ccname)];
                    outfile_handles{k} = fopen([chs_info.output_name{ix} ch_ext],'w');
                end
           end
        end

        new_files(fi).first_sample = 1;
    else
        new_files(fi).first_sample = new_files(fi-1).lts + new_files(fi-1).first_sample;
    end
    if isempty(parsed_chs)
        disp('Without channels to parse.')
        return
    end
   
    %total lenght adding the zeros from Timestamp
    lts = hFile.TimeSpan/(30000/sr);
    new_files(fi).lts = lts;

    N = lts;   % total data points
    num_segments = ceil(N/samples_per_channel);
    samples_per_segment = min(samples_per_channel,N);
    fprintf('Data will be processed in %d segments of %d samples each.\n',num_segments,samples_per_segment)
    
%     min_valid_val = zeros([nchan,1]);
%     max_valid_val = zeros([nchan,1]);
%     for i = 1:nchan
%         [~, nsAnalogInfo] = ns_GetAnalogInfo(hFile, i); %ns5: min -8191 max 8191 resol 0.25; nf3 min 0 max 17920 resol 1 
%         min_valid_val(i) = nsAnalogInfo.MinVal/nsAnalogInfo.Resolution;
%         max_valid_val(i) = nsAnalogInfo.MaxVal/nsAnalogInfo.Resolution;
%     end
    
    for j=1:num_segments
        ini = (j-1)*samples_per_segment+1;
        fin = min(j*samples_per_segment,N);
        tcum = tcum + toc;  % this is because openNSx has a tic at the beginning
        [~, Data] = ns_GetAnalogDataBlock(hFile, electrodes_row, ini, fin-ini+1, 'unscale'); %ns5 Data is int16, ns3 Data is single
        for i = 1:nchan
            if ~isempty(outfile_handles{i}) %channels with empty outfile_handles{i} are not selected
                if sr==30000
%                     pak_lost = Data(:,i)<min_valid_val(i);
                    pak_lost = Data(:,i) == intmin('int16');
                    Data(pak_lost,i)=0;
                    chs_info.pak_list(i) = chs_info.pak_list(i)+sum(pak_lost);  
                else
%                     pak_lost = Data(:,i) > 3e+38; % maximum value for a single-precision floating-point number 
                    pak_lost = Data(:,i) == realmax('single');
                    Data(pak_lost,i)=0;
                    chs_info.pak_list(i) = chs_info.pak_list(i)+sum(pak_lost);
                end
                Data_ch = double(Data(:,i));
                if sr~=30000
                    chs_info.dc(i)=(max(Data_ch)+min(Data_ch))/2;
                    Data_ch=Data_ch- chs_info.dc(i);
                    chs_info.conversion(i) = max(abs(Data_ch))/double(intmax('int16'));
                    Data_ch = round(Data_ch/chs_info.conversion(i));
                end
                fwrite(outfile_handles{i},int16(Data_ch),'int16');
            end
        end
        fprintf('Segment %d out of %d processed. ',j,num_segments)
    end
    
    tcum = tcum + toc;
    fprintf('Total time spent in parsing the data was %s secs.\n',num2str(tcum, '%0.1f')); 
end
fclose('all');



if isempty(metadata)
    files = [];
    NSx = [];
else
    NSx = metadata.NSx;
    files = metadata.files;
end
lts = sum([new_files(:).lts]);
fprintf('%d data points written per channel\n',lts)


for ci = 1:length(new_channel_id)
    ch = new_channel_id(ci);
    elec_id = parsed_chs(ci);
    if sr == 7500
        repetead = arrayfun(@(x) (x.chan_ID==ch) && endsWith(x.label,hifreq_raw),NSx);
    else
        repetead = arrayfun(@(x) (x.chan_ID==ch) && (x.sr==sr),NSx);
    end
    if isempty(repetead) || sum(repetead)==0
        pos = length(NSx)+1;
    else
        pos = find(repetead);
    end
        ix = chs_info.id==elec_id;
%         ix = find(ix,1);
        NSx(pos).chan_ID = ch;
        NSx(pos).conversion = chs_info.conversion(ix);
        NSx(pos).dc = chs_info.dc(ix);
        NSx(pos).label = chs_info.label{ix};
        NSx(pos).bundle = get_bundle(chs_info.macro{ix});
%         NSx(pos).is_micro = NSx(pos).label(1)=='m';
        NSx(pos).unit = chs_info.unit{ix};
        NSx(pos).electrode_ID = elec_id;
        NSx(pos).which_system = 'RIPPLE';
        NSx(pos).ext = ch_ext;
        NSx(pos).lts = lts;
        NSx(pos).filename = filenames;
        NSx(pos).sr = sr;
        NSx(pos).output_name = chs_info.output_name{ix};
        NSx(pos).pak_list =chs_info.pak_list(ix);
        NSx(pos).is_micro =strcmp(NSx(pos).unit,'uV') && NSx(pos).sr==30000;
end

for i = 1:length(new_files)
    repetead = arrayfun(@(x) strcmp(x.name,new_files(i).name),files);
    if isempty(repetead) || sum(repetead)==0
        pos = length(files)+1;
    else
        pos = find(repetead);
    end
    files(pos).name = new_files(i).name;
    files(pos).first_sample = new_files(i).first_sample;
    files(pos).lts = new_files(i).lts;
end
freq_priority=[30000, 2000, 7500, 10000, 1000, 500];
save(metadata_file, 'NSx','files','freq_priority','Date_Time')
% % custom_path.rm()
end

function bundle=get_bundle(label)
    tt = regexp(label,'\d+','once');
    if ~isempty(tt)
        bundle = label(1:tt-1);
    else
        bundle = label;
    end    
%     macro = regexp(label,'^\w*(?=(\d* raw$))','match');
%     if ~isempty(macro)
%         macro = macro{1};
%     else
%         macro = label;
%     end
end
