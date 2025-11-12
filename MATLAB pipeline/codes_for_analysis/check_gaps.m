function check_gaps(foldernames)

clearvars

%%
% Define the path where the codes emu repository is located
[~,name] = system('hostname');
if contains(name,'BEH-REYLAB'), dir_base = '/home/user/share/codes_emu';
elseif contains(name,'TOWER-REYLAB') || contains(name,'RACK-REYLAB'),
    %     current_user = 'sofiad';  % replace with appropriate user name
    current_user = getenv('USER');    dir_base = sprintf('/home/%s/Documents/GitHub/codes_emu',current_user);
elseif contains(name,'NSRG-HUB-15446'), dir_base = 'D:\codes_emu'; % Hernan's desktop
end

addpath(dir_base);
custompath = reylab_custompath({'wave_clus_reylab','NPMK','codes_for_analysis','mex','useful_functions','neuroshare' });


if ~exist('dirmain','var')
    dirmain = pwd;
end

% addpath(genpath([dirmain '_pic'])) % srtimuli folder
set(groot,'defaultaxesfontsmoothing','off')
set(groot,'defaultfiguregraphicssmoothing','off')
set(groot,'defaultaxestitlefontsizemultiplier',1.1)
set(groot,'defaultaxestitlefontweight','normal')

%%

micros =  dir('**/*.ns5');
filens5= {micros.name};

macros = dir('**/*.nf3');
filenf3= {macros.name};

fprintf('MICROS \n');
for k = 01:length(micros)
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

    metadata_file = fullfile(pwd, 'NSx.mat');
    if exist(metadata_file,'file')
        metadata = load(metadata_file);
    else
        metadata = [];
    end

    tic
    [ns_status, hFile] = ns_OpenFile(filens5{1}, 'single');

    hFileModified = hFile;
    numEntities = length(hFile.Entity);
    keywordsToRemove = {'Photo_analog', 'MicL', 'MicR','analog'};
    stringToRemove = 'ref';

    for i = numEntities:-1:1
        currentLabel = hFile.Entity(i).Label;

        if any(contains(keywordsToRemove, currentLabel))
            hFileModified.Entity(i) = [];
        elseif contains(currentLabel, stringToRemove)
            hFileModified.Entity(i) = [];
        end
    end

    fid = fopen(filens5{1}, 'rb');
    fseek(fid, 294, -1);
    %     Date = fread(fid, 8, 'uint16');

    if strcmp(ns_status,'ns_FILEERROR')
        error('Unable to open file: %s',filename)
    end

    switch hFile.FileInfo.Type(3)
        case '1'
            sr = 500;
        case '2'
            sr = 1000;
        case '3'
            sr = 2000;
        case '4'
            sr = 10000;
        case {'5' ,'6'}
            sr = 30000;
        otherwise
            error('ERROR: %s file type not supported',hFile.FileInfo.Type)
    end

    nchan = size(hFile.Entity,2);   % number of channels
    samples_per_channel = ceil(max_memo/(nchan*2));
    if fi == 1
        %get info about channels in nsx file
        chs_info = struct();
        chs_info.unit = cellfun(@(x) deblank(x),{hFile.Entity.Units},'UniformOutput',false);
        chs_info.label = cellfun(@(x) deblank(x),{hFile.Entity.Label},'UniformOutput',false)';
        chs_info.conversion = cell2mat({hFile.Entity.Scale});
        chs_info.id = cell2mat({hFile.Entity.ElectrodeID});
        chs_info.pak_list = 0*chs_info.id;
        chs_info.dc= double(0*chs_info.id);
        chs_info.macro = chs_info.label;
        micros = cellfun(@(x) strcmp(x,'uV'),chs_info.unit);
        if ~isempty(macro)
            chs_info.macro(micros) = arrayfun(@(x) macro{ceil(x/9)},find(micros),'UniformOutput',false);
        end

        [~,~,fext] = fileparts(filename);
        fext = lower(fext(2:end));
        nsx_ext = fext(end);
        ch_ext = ['.NC' nsx_ext];
        if ~exist('channels','var') || isempty(channels)
            channels = hFile.FileInfo.ElectrodeList;
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
    end

    remove_channels = [];
    for i=1:nchan
        currentLabel = hFile.Entity(i).Label;
        if contains(currentLabel, {'ref','Photo_analog', 'MicL', 'MicR','analog'})
            remove_channels = [remove_channels, i];
        end
    end

    lts = hFile.TimeSpan/(30000/sr);
    %     new_files(fi).lts = lts;

    N = lts;   % total data points
    num_segments = ceil(N/samples_per_channel);
    samples_per_segment = 5400000;
    %             samples_per_segment = min(samples_per_channel,N);
    fprintf('Gaps run %d: Data will be processed in %d segments of %d samples each.\n',k,num_segments,samples_per_segment)

    for j=1:num_segments
        %                 ini = (j-1)*samples_per_segment+1;
        ini = N-samples_per_segment;
        %                 fin = min(j*samples_per_segment,N);
        fin = N;
        tcum = tcum + toc;  % this is because openNSx has a tic at the beginning
        [~, Data] = ns_GetAnalogDataBlock(hFile, [1:nchan], ini, fin-ini, 'scale'); %ns5 Data is int16, ns3 Data is single
        Data = Data.^2;
        m5 = width(Data);
        for i = 1:nchan
            x = mean(Data(:,i));
            x = log10(x+1);
            metadata{i,1} = x;
        end
    end

    if ~exist('data_micros','var')
        data_micros = [metadata];
        xtick = [1];
    else
        %         m2 = [metadata; metadata2]
        data_micros = horzcat(data_micros, [metadata]);
        xtick = [1:i];
    end

end

    function result = remove_substrings(input_str, substrings)
        result = input_str;
        for i = 1:length(substrings)
            result = strrep(result, substrings{i}, '');
        end
    end

micros = m5;
data_micros2 = [];
for i = 1:m5
    if ~ismember(i, remove_channels)
        data_micros2 = [data_micros2; data_micros(i, :)];
    else
        micros = micros-1;
    end
end

label_micros = cell(1,micros);
for i = 1:micros
    label_str = hFileModified.Entity(i).Label;
    label_micros{i} = remove_substrings(label_str, {' raw'});
end

ylabels_micros = label_micros;
yl_unique = cell(size(label_micros));
seen_names = {};
splitter = [];
for i = 1:length(label_micros)
    full_name = label_micros{i};
    name = full_name(1:end-2); % Remove the last two digits
    if ~iscell(seen_names)
        seen_names = cell(seen_names);
    end

    if ~ismember(name,seen_names)
        if i == 1
            seen_names = {name};
            yl_unique{i} = name;
        else
            seen_names = [seen_names, {name}];
            yl_unique{i} = name;
            splitter = [splitter,i-1];
        end
    end
end
label_micros = yl_unique;

ytick = 1:m5;

micros = cell2mat(data_micros2);
[num_filas, num_colum] = size(micros);

figure
imagesc(micros)
colorbar
set(gcf,'units','normalized','outerposition',[0 0 1 1], 'Visible', 'off') 

for i = 1:num_filas - 1
    if ismember(i,splitter)
        yline(i + 0.5, 'k',LineWidth=1.5); 
    else
        yline(i + 0.5, ':k',LineWidth=0.01); 
    end
end

for j= 1:num_colum -1
    xline(j + 0.5, 'k',LineWidth=1.0);
end

xl = cell(1, num_colum);

for i = 1:num_colum
    run_str = sprintf('%02d', i);
    xl{i} = run_str;
end

title('Micros')
set(gca,'XTick',xtick,'XTickLabel',xl,'TickLength',[0, 0])
set(gca,'YTick',ytick,'YTickLabel',label_micros,'TickLength',[0, 0])
xlabel('Gaps run');

saveas(gcf, 'micros.png');
%%
% MACROS

fprintf('MACROS \n');
for j = 01:length(macros)

    metadata2=[];
    tic
    [ns_status, hFile2] = ns_OpenFile(filenf3{1}, 'single');

    hFileModified2 = hFile2;
    numEntities = length(hFile2.Entity);
    stringToRemove = {'EKG','-Z','-COM'};

    for i = numEntities:-1:1
        currentLabel = hFile2.Entity(i).Label;

        if contains(currentLabel, stringToRemove)
            hFileModified2.Entity(i) = [];
        end
    end

    fid = fopen(filenf3{1}, 'rb');
    fseek(fid, 294, -1);

    if strcmp(ns_status,'ns_FILEERROR')
        error('Unable to open file: %s',filename)
    end

    switch hFile2.FileInfo.Type(3)
        case '1'
            sr = 500;
        case '2'
            sr = 1000;
        case '3'
            sr = 2000;
        case '4'
            sr = 10000;
        case {'5' ,'6'}
            sr = 30000;
        otherwise
            error('ERROR: %s file type not supported',hFile2.FileInfo.Type)
    end

    nchan = size(hFile2.Entity,2);   % number of channels
    samples_per_channel = ceil(max_memo/(nchan*2));
    if fi == 1
        %get info about channels in nsx file
        chs_info = struct();
        chs_info.unit = cellfun(@(x) deblank(x),{hFile2.Entity.Units},'UniformOutput',false);
        chs_info.label = cellfun(@(x) deblank(x),{hFile2.Entity.Label},'UniformOutput',false)';
        chs_info.conversion = cell2mat({hFile2.Entity.Scale});
        chs_info.id = cell2mat({hFile2.Entity.ElectrodeID});
        chs_info.pak_list = 0*chs_info.id;
        chs_info.dc= double(0*chs_info.id);
        chs_info.macro = chs_info.label;
        micros = cellfun(@(x) strcmp(x,'uV'),chs_info.unit);
        if ~isempty(macro)
            chs_info.macro(micros) = arrayfun(@(x) macro{ceil(x/9)},find(micros),'UniformOutput',false);
        end

        [~,~,fext] = fileparts(filename);
        fext = lower(fext(2:end));
        nsx_ext = fext(end);
        ch_ext = ['.NC' nsx_ext];
        if ~exist('channels','var') || isempty(channels)
            channels = hFile2.FileInfo.ElectrodeList;
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
    end

    for i=1:nchan
        currentLabel = hFile2.Entity(i).Label;
        stringToRemove = {'EKG','-Z','-COM'};
        if contains(currentLabel, stringToRemove)
            remove_channels = [remove_channels, i+m5];
        end
    end

    lts = hFile2.TimeSpan/(30000/sr);
    %     new_files(fi).lts = lts;

    N = lts;   % total data points
    num_segments = ceil(N/samples_per_channel);
    %             samples_per_segment = min(samples_per_channel,N);
    samples_per_segment = 360000;
    fprintf('Gaps run %d: Data will be processed in %d segments of %d samples each.\n',j,num_segments,samples_per_segment)

    for j=1:num_segments
        %                 ini = (j-1)*samples_per_segment+1;
        ini = N-samples_per_segment;
        %                 fin = min(j*samples_per_segment,N);
        fin = N;
        tcum = tcum + toc;  % this is because openNSx has a tic at the beginning
        [~, Data2] = ns_GetAnalogDataBlock(hFile2, [1:nchan], ini, fin-ini+1, 'scale'); %ns5 Data is int16, ns3 Data is single
        Data2 = Data2.^2;
        m3 = width(Data2);
        for i = 1:nchan
            x = mean(Data2(:,i))+1;
            x = log10(x+1);
            metadata2{i,1} = x;
        end
    end

    if ~exist('data_macros','var')
        data_macros = [metadata2];
        xtick = [1];
    else
        data_macros = horzcat(data_macros, [metadata2]);
        xtick = [1:i];
    end
end 

macros = m3;
data_macros2 = [];
for i = 1:m3
    if ~ismember(i, remove_channels)
        data_macros2 = [data_macros2; data_macros(i, :)];
    else
        macros = macros-1;
    end
end

label_macros = cell(1,macros);
for i = 1:macros
    label_str = hFileModified2.Entity(i).Label;
    label_macros{i} = remove_substrings(label_str, {' hi-res'});
end

ylabels_macros = label_macros;
yl_unique = cell(size(label_macros));
seen_names = {};
splitter2 = [];
for i = 1:length(label_macros)
    full_name = label_macros{i};
    name = full_name(1:end-2); % Remove the last two digits
    if ~iscell(seen_names)
        seen_names = cell(seen_names);
    end

    if ~ismember(name,seen_names)
        if i == 1
            seen_names = {name};
            yl_unique{i} = name;
        else
            seen_names = [seen_names, {name}];
            yl_unique{i} = name;
            splitter2 = [splitter2,i-1];
        end
    end
end
label_macros = yl_unique;

ytick = 1:m3;

macros = cell2mat(data_macros2);
[num_filas, num_colum] = size(macros);

figure
imagesc(macros)
colorbar
set(gcf,'units','normalized','outerposition',[0 0 1 1], 'Visible', 'off') 

for i = 1:num_filas - 1
    if ismember(i,splitter2)
        yline(i + 0.5, 'k',LineWidth=1.5); 
    else
        yline(i + 0.5, ':k',LineWidth=0.01); 
    end
end

for j= 1:num_colum -1
    xline(j + 0.5, 'k',LineWidth=1.0);
end

xl = cell(1, num_colum);

for i = 1:num_colum
    run_str = sprintf('%02d', i);
    xl{i} = run_str;
end

title('Macros')
set(gca,'XTick',xtick,'XTickLabel',xl,'TickLength',[0, 0])
set(gca,'YTick',ytick,'YTickLabel',label_macros,'TickLength',[0, 0])
xlabel('Gaps run');

saveas(gcf, 'macros.png');

end









