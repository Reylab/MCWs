function parse_NSx_Gemi_brief(nsp_filenames,macro)
addpath(genpath(fullfile(fileparts(mfilename('fullpath')),'..','..','NPMK-master_Gemini','NPMK'))) % always / for this trick

% addpath(genpath('..\..\..\NPMK-master_Gemini\NPMK'))
% hours_offset = 6; %for MCW
expr2remove = '-\d+$';
% which_nsp = [];
nsp=[];

% for fi = 1:length(nsp_filenames)
fi = 1;
filename= nsp_filenames{fi};
new_files(fi).name = filename;

tic
nsx_file = openNSx(filename, 'read','report');
toc

sr = nsx_file.MetaTags.SamplingFreq;   % sampling rate
nchan = nsx_file.MetaTags.ChannelCount;   % number of channels
chs_info = struct();
chs_info.unit = cellfun(@(x) max_analog2unit(x),{nsx_file.ElectrodesInfo.MaxAnalogValue},'UniformOutput',false)';
chs_info.label = cellfun(@(x) deblank(x),{nsx_file.ElectrodesInfo.Label},'UniformOutput',false)';
chs_info.conversion = (double(cell2mat({nsx_file.ElectrodesInfo.MaxAnalogValue}))./double(cell2mat({nsx_file.ElectrodesInfo.MaxDigiValue})))';
chs_info.id = cell2mat({nsx_file.ElectrodesInfo.ElectrodeID});
chs_info.pak_list = 0*chs_info.id;
chs_info.dc= double(0*chs_info.id);
chs_info.macro = chs_info.label;
micros = cellfun(@(x) strcmp(x,'uV'),chs_info.unit);
if exist('macro','var') && ~isempty(macro)
    %             chs_info.macro(micros) = arrayfun(@(x) macro{ceil(x/8)},find(micros),'UniformOutput',false);
    chs_info.macro(1:size(macro,1)*8) = arrayfun(@(x) macro{ceil(x/8)},find(micros(1:size(macro,1)*8)),'UniformOutput',false);
end

outfile_handles = cell(1,nchan); %some will be empty
[~,~,fext] = fileparts(filename);
fext = lower(fext(2:end));
nsx_ext = fext(end);
ch_ext = ['.NC' nsx_ext];
%         if ~exist('channels','var') || isempty(channels)
channels = nsx_file.MetaTags.ChannelID;
%         end
parsed_chs = [];
new_channel_id = [];

for i = 1:nchan
    c = nsx_file.MetaTags.ChannelID(i);
    if ismember(c,channels)
        ccname = c;
        parsed_chs(end+1) = c;
                new_channel_id(end+1) = ccname;
        ix = chs_info.id==c;
        %                 output_name = chs_info.label{ix};
        output_name = chs_info.macro{ix};
        outn_i = regexp(output_name,expr2remove);
        if ~isempty(outn_i) && outn_i>1
            output_name = output_name(1:outn_i-1);
        end
        chs_info.output_name{ix} = [output_name '_' num2str(ccname)];
        outfile_handles{i} = fopen([chs_info.output_name{ix} ch_ext],'w');
    end
end
new_files(fi).first_sample = 1;
new_files(fi).trim4sinc = 0;

%     DataPoints = nsx_file.MetaTags.DataPoints;
lts = nsx_file.MetaTags.DataPoints;
new_files(fi).lts = lts;
    init_cell = 1;
if length(lts)>1
    fprintf('NSx.MetaTags.Timestamp: %s', formatvector(nsx_file.MetaTags.Timestamp,'.f'))
    fprintf('NSx.MetaTags.DataPoints: %s', formatvector(nsx_file.MetaTags.DataPoints,'.f'))
    fprintf('NSx.MetaTags.DataDurationSec: %s', formatvector(nsx_file.MetaTags.DataDurationSec,'.5f'))
    if init_cell==1
        error('need to define which cell needs to be saved')
    end
end
% new_files(fi).which_cells = init_cell:length(nsx_file.MetaTags.DataPoints);
new_files(fi).which_cells = init_cell;
zeros2add = 0;
accum_lts = 0; %count written samples to solve pauses
% if (accum_lts + zeros2add + size(nsx_file.Data,2)) > lts
%                 data_end = lts - accum_lts - zeros2add;
%             else
lts = nsx_file.MetaTags.DataPoints(init_cell);
% data_ok =nsx_file.Data(new_files(fi).which_cells);
if iscell(nsx_file.Data) 
    data_ok =nsx_file.Data{init_cell};
else 
    data_ok =nsx_file.Data;
end
data_end = size(data_ok,2);
%             end
for i = 1:nchan
    if ~isempty(outfile_handles{i}) %channels with empty outfile_handles{i} are not selected
        %                     if (j==1) && (nsx_file.MetaTags.Timestamp>0)
        %                         fwrite(outfile_handles{i},zeros(zeros2add,1,'int16'),'int16');
        %                     end
        Data_ch = data_ok(i,1:data_end);
        % BRK IS ALWAYS INT16!!!
%         if sr~=30000
% %             Data_ch = Data_ch*chs_info.conversion(i);
%             Data_ch = double(Data_ch)*chs_info.conversion(i);
%             chs_info.dc(i)=(max(Data_ch)+min(Data_ch))/2;
%             Data_ch=Data_ch- chs_info.dc(i);
%             chs_info.conversion(i) = max(abs(Data_ch))/double(intmax('int16'));
%             Data_ch = round(Data_ch/chs_info.conversion(i));
%         end
        fwrite(outfile_handles{i},Data_ch,'int16');      
            
    end
end

fclose('all');

metadata_file = fullfile(pwd, 'NSx.mat');
if exist(metadata_file,'file')
    metadata = load(metadata_file);
else
    metadata = [];
end

Date = nsx_file.MetaTags.DateTimeRaw;
tUTC = datetime([Date(1:2) Date(4:7)],'TimeZone','America/Chicago');
[dt,dst] = tzoffset(tUTC);
Start_Time = tUTC+dt;

% Start_Time = datetime([Date(1:2) Date(4:7)])-hours(hours_offset);
% Rec_length_sec = nsx_file.MetaTags.DataDurationSec;
Rec_length_sec = nsx_file.MetaTags.DataDurationSec(init_cell);
End_Time = Start_Time + seconds(Rec_length_sec);
Date_Time = [Start_Time;End_Time];

if isempty(metadata)
    files = [];
    NSx = [];
else
    NSx = metadata.NSx;
    files = metadata.files;
end

for ci = 1:length(new_channel_id)
    ch = new_channel_id(ci);
    elec_id = parsed_chs(ci);
    repetead = arrayfun(@(x) (x.chan_ID==ch) && (x.sr==sr) ,NSx);
    if isempty(repetead) || sum(repetead)==0
        pos = length(NSx)+1;
    else
        pos = find(repetead);
    end
    ix = chs_info.id==elec_id;
    NSx(pos).chan_ID = ch;
    NSx(pos).conversion = chs_info.conversion(ix);
    NSx(pos).dc = chs_info.dc(ix);
    NSx(pos).label = chs_info.label{ix};
    %         NSx(pos).is_micro = NSx(pos).label(1)=='m';
    if sr==30000 
        NSx(pos).is_micro = true;
    else 
        NSx(pos).is_micro = false;
    end
    NSx(pos).unit = chs_info.unit{ix};
    NSx(pos).electrode_ID = elec_id;
    NSx(pos).nsp = nsp;
    NSx(pos).which_system = 'BRK-Gem';
    NSx(pos).ext = ch_ext;
    NSx(pos).lts = lts;
    NSx(pos).filename = nsp_filenames;
    NSx(pos).sr = sr;
    NSx(pos).output_name = chs_info.output_name{ix};
    NSx(pos).pak_list =chs_info.pak_list(ix);

    %NSx(pos).bundle = chs_info.macro{ix};
    NSx(pos).bundle = get_bundle(chs_info.macro{ix});

    %         macro_i = regexp(chs_info.label{ix},'\d+-\d+$','start','once');
    %         if isempty(macro_i) || macro_i<2
    %             NSx(pos).macro = chs_info.output_name{ix};
    %         else
    %             NSx(pos).macro = chs_info.output_name{ix}(1:macro_i-1);
    %         end
end

if ~isempty(new_channel_id)
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
        files(pos).which_nsp = nsp;
        files(pos).trim4sinc =  new_files(i).trim4sinc;
%         files(pos).which_cells = init_cell:(length(nsx_file.MetaTags.DataPoints));
        files(pos).which_cells = init_cell;
    end
    %     metadata.NSx = NSx;
    %     metadata.files = files;
end

freq_priority=[30000, 2000, 10000, 1000, 500];
metadata_file = fullfile(pwd, 'NSx.mat');
save(metadata_file, 'NSx','files','freq_priority','Date_Time')

% rmpath(genpath('C:\Users\hgrey\OneDrive - mcw.edu\BRK\NPMK-master_Gemini\NPMK'))
rmpath(genpath(fullfile(fileparts(mfilename('fullpath')),'..','..','NPMK-master_Gemini','NPMK'))) % always / for this trick

end

function unit = max_analog2unit(x)
switch x
    case 5000
        unit='mV';
    case 8191
        unit ='uV';
end
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

function fv = formatvector(v,f)
    fv=sprintf(['[' repmat(['%' f ', '], 1, numel(v)-1), ['%' f ']\n']  ],v);
end
