function parse_NSx(filenames,which_nsp ,max_memo_GB,macro,overwrite,channels)
% This code requires the file openNSx.m, from the NPMK, in the path. You can download the download NPMK from https://github.com/BlackrockMicrosystems/NPMK/releases
% filenames is the full path to the nsx file to be parsed. If a cell array
% of strings is passed, it will read all the files and concatenate them in
% a single recording.
% which_nsp can be empty (assumes there was only one NSP used in the
% recording), 1 or 2 (it will add 1000 or 2000 to the channel number so
% they can be distinguished after parsing or a vector with the nsp associated
% with each file (50% from nsp 1 and 50% from nsp 2). If there were multiople cells in the recording, you will 
% be prompted to choose the starting cell.
% max_memo_GB is an idea of the number of GB allocated for the data to be
% stored in RAM, so it is used to compute the number of segments in which
% the data should be split for processing

expr2remove = '-\d+$';

if ~exist('overwrite','var') || isempty(overwrite)
    overwrite = false;
end

if ~exist('macro','var')
    macro = [];
end

if ~exist('which_nsp','var')
    which_nsp = [];
end

if numel(which_nsp)>1
    [init_cells, maxlts, trims4sinc] = select_init_cells(filenames, which_nsp);
end

with_memory=true;
try
	memory;
catch
	with_memory=false;
end
if with_memory
	[userview,systemview] = memory;
	memo_avaible = floor(systemview.PhysicalMemory.Available*0.80);
	if exist('max_memo_GB','var') && ~isempty(max_memo_GB)
        max_memo = max_memo_GB*(1024)^3;
		if max_memo > memo_avaible
			error('max_memo_GB > 80% of Physical Memory Available')
		end
	else
		max_memo = memo_avaible;
	end
else
    if ~exist('max_memo_GB','var') || isempty(max_memo_GB)
        error('You have to specify the parameter ''max_memo_GB''')
    end
	max_memo = max_memo_GB*(1024)^3;
end

tcum=0;
if ischar(filenames)
    filenames = {filenames};
end


metadata_file = fullfile(pwd, 'NSx.mat');
if exist(metadata_file,'file')
    metadata = load(metadata_file);
else
    metadata = [];
end

for nsp = unique(which_nsp)
if numel(which_nsp)>1
    nsp_filenames = filenames(which_nsp==nsp);
else
    nsp_filenames = filenames;
end
for fi = 1:length(nsp_filenames)
    filename= nsp_filenames{fi};
    new_files(fi).name = filename;
    if length(filename)<3 || (~strcmpi(filename(2:3),':\') && ...
                     ~strcmpi(filename(1),'/') && ...
                     ~strcmpi(filename(2),'/') && ...
                     ~strcmpi(filename(1:2), '\\')&& ~strcmpi(filename(2:3),':/'))
        filename= [pwd filesep filename];
    end
    
    nsx_file = openNSx(filename, 'report','noread');
    if isnumeric(nsx_file) && nsx_file==-1
        error('Unable to open file: %s',filename)
    end
    
    sr = nsx_file.MetaTags.SamplingFreq;   % sampling rate
    nchan = nsx_file.MetaTags.ChannelCount;   % number of channels
    samples_per_channel = ceil(max_memo/(nchan*2));
    if fi == 1
        %get info about channels in nsx file
        chs_info = struct();
        chs_info.unit = cellfun(@(x) max_analog2unit(x),{nsx_file.ElectrodesInfo.MaxAnalogValue},'UniformOutput',false)';
        chs_info.label = cellfun(@(x) deblank(x),{nsx_file.ElectrodesInfo.Label},'UniformOutput',false)';
        chs_info.conversion = (double(cell2mat({nsx_file.ElectrodesInfo.MaxAnalogValue}))./double(cell2mat({nsx_file.ElectrodesInfo.MaxDigiValue})))';
        chs_info.id = cell2mat({nsx_file.ElectrodesInfo.ElectrodeID});

        chs_info.macro = chs_info.label;
        micros = cellfun(@(x) strcmp(x,'uV'),chs_info.unit);
        if ~isempty(macro)
%             chs_info.macro(micros) = arrayfun(@(x) macro{ceil(x/8)},find(micros),'UniformOutput',false);
            chs_info.macro(1:size(macro,1)*8) = arrayfun(@(x) macro{ceil(x/8)},find(micros(1:size(macro,1)*8)),'UniformOutput',false);
        end
        
        outfile_handles = cell(1,nchan); %some will be empty
        [~,~,fext] = fileparts(filename);
        fext = lower(fext(2:end));
        nsx_ext = fext(end);
        ch_ext = ['.NC' nsx_ext];
        if ~exist('channels','var') || isempty(channels)
            channels = nsx_file.MetaTags.ChannelID;
        end
        parsed_chs = [];
        new_channel_id = [];
        for i = 1:nchan
            c = nsx_file.MetaTags.ChannelID(i);
            if ismember(c,channels)
                ccname = c;
                if ~isempty(nsp)
                    ccname = ccname + 1000*nsp;
                end
                if ~isempty(metadata) %NSx file in current folder
                    repetead = arrayfun(@(x) (x.chan_ID==ccname) && (x.sr==sr) ,metadata.NSx);
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
    else
        new_files(fi).first_sample = new_files(fi-1).lts + new_files(fi-1).first_sample;
    end
    if isempty(parsed_chs)
        disp('Without channels to parse.')
        break
    end
    DataPoints = nsx_file.MetaTags.DataPoints;
    if numel(which_nsp)>1 %for concatenation with multiple nsp
        lts = maxlts(nsp,fi);
        init_cell = init_cells(nsp,fi);
        new_files(fi).trim4sinc = trims4sinc(nsp,fi);
    else
        if length(DataPoints)>1 && ~isempty(nsp)
            fprintf('\n\n')
            fprintf('###################\n')
            fprintf('###################\n')
            fprintf('NSx.MetaTags.Timestamp: %s', formatvector(nsx_file.MetaTags.Timestamp,'.f'))
            fprintf('NSx.MetaTags.DataPoints: %s', formatvector(nsx_file.MetaTags.DataPoints,'.f'))
            fprintf('NSx.MetaTags.DataDurationSec: %s', formatvector(nsx_file.MetaTags.DataDurationSec,'.5f'))
            init_cell = [];
            while isempty(init_cell) || (init_cell> length(DataPoints)) || (init_cell<1)
                if ~isempty(init_cell)
                    warning('wrong answer')
                end
                init_cell = input(sprintf('From which cell number you want to save data? From 1 to %d: ',length(DataPoints)));
            end
        else
            init_cell = 1;
        end

        %total lenght adding the zeros from Timestamp
        Timestamp = round(nsx_file.MetaTags.Timestamp(init_cell:end)*nsx_file.MetaTags.SamplingFreq/nsx_file.MetaTags.TimeRes);
        cellslts = zeros(numel(nsx_file.MetaTags.Timestamp(init_cell:end)),1);
        cellslts(1) = floor(nsx_file.MetaTags.DataPoints(init_cell)) + Timestamp(1);
        
        for ci=2:numel(cellslts)
            cellslts(ci)=Timestamp(ci)-sum(cellslts(1:ci-1))+ floor(nsx_file.MetaTags.DataPoints(init_cell+ci-1));
        end
        lts = sum(cellslts);
    
        new_files(fi).trim4sinc = 0;
    end
    
    new_files(fi).lts = lts;
    new_files(fi).which_cells = init_cell:length(DataPoints);
    csDataPoints = [0 cumsum(DataPoints)];
    accum_lts = 0; %count written samples to solve pauses
    for part = init_cell:length(DataPoints)
        N = floor(DataPoints(part));   % total data points
        num_segments = ceil(N/samples_per_channel);
        fprintf('Data will be processed in %d segments of %d samples each.\n',num_segments,min(samples_per_channel,N))
        for j=1:num_segments
            ini = (j-1)*samples_per_channel+1+floor(csDataPoints(part));
            fin = min(j*samples_per_channel,N)+floor(csDataPoints(part));
            tcum = tcum + toc;  % this is because openNSx has a tic at the beginning
            nsx_file = openNSx('read',filename,['t:' num2str(ini) ':' num2str(fin)],'nozeropad'); % if used without time windows, it add zeros for the very first cell, but may be not for the others.
            if j==1
                zeros2add = round(nsx_file.MetaTags.Timestamp*nsx_file.MetaTags.SamplingFreq/nsx_file.MetaTags.TimeRes) - accum_lts;
            else
                zeros2add = 0;
            end
            if (accum_lts + zeros2add + size(nsx_file.Data,2)) > lts
                data_end = lts - accum_lts - zeros2add;
            else
                data_end = size(nsx_file.Data,2);
            end
            for i = 1:nchan
                if ~isempty(outfile_handles{i}) %channels with empty outfile_handles{i} are not selected
                    if (j==1) && (nsx_file.MetaTags.Timestamp>0)
                        fwrite(outfile_handles{i},zeros(zeros2add,1,'int16'),'int16');
                    end
                    fwrite(outfile_handles{i},nsx_file.Data(i,1:data_end),'int16');
                end
            end
            accum_lts = accum_lts + zeros2add + size(nsx_file.Data,2); 
            fprintf('Segment %d out of %d processed. ',j,num_segments)
            if j==1
                fprintf('Zeros added = %d. ',zeros2add);
            end
            fprintf('Data Point Read = %d \n',size(nsx_file.Data,2));
        end
    end
    
    tcum = tcum + toc;
    fprintf('Total time spent in parsing the data was %s secs.\n',num2str(tcum, '%0.1f')); 
    if length(DataPoints)>1 && isempty(nsp)
        warning('Automatically merged %d cells of data.', length(DataPoints) )
        fprintf('NSx.MetaTags.Timestamp: %s', formatvector(nsx_file.MetaTags.Timestamp,'.f'))
        fprintf('NSx.MetaTags.DataPoints: %s', formatvector(nsx_file.MetaTags.DataPoints,'.f'))
    end
end
fclose('all');



if isempty(metadata)
    files = [];
    NSx = [];
else
    NSx = metadata.NSx;
    files = metadata.files;
end

if ~isempty(new_channel_id)
    lts = sum([new_files(:).lts]);
    fprintf('%d data points written per channel\n',lts)
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
        NSx(pos).label = chs_info.label{ix};
        NSx(pos).unit = chs_info.unit{ix};
        NSx(pos).electrode_ID = elec_id;
        NSx(pos).nsp = nsp;
        NSx(pos).which_system = 'BRK';
        NSx(pos).ext = ch_ext;
        NSx(pos).lts = lts;
        NSx(pos).filename = nsp_filenames;
        NSx(pos).sr = sr;
        NSx(pos).output_name = chs_info.output_name{ix};
        macro_i = regexp(chs_info.label{ix},'\d+-\d+$','start','once');
        if isempty(macro_i) || macro_i<2
            NSx(pos).macro = chs_info.output_name{ix};
        else
            NSx(pos).macro = chs_info.output_name{ix}(1:macro_i-1);
        end
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
        files(pos).which_cells = init_cell:(length(DataPoints));
    end
    metadata.NSx = NSx;
    metadata.files = files;
end
end
freq_priority=[30000, 2000, 10000, 1000, 500];
save(metadata_file, 'NSx','files','freq_priority')
if exist('check_parser_sync.m','file')
    if numel(which_nsp)>1
        sincs = cellfun(@(x) strcmp(x,'RecordingSync'),{NSx.label});
        sinc_nsp1 = find(sincs & [NSx.nsp]==1);
        sinc_nsp2 = find(sincs & [NSx.nsp]==2);
        if ~isempty(sinc_nsp1) && ~isempty(sinc_nsp2)
            disp('--Sync files found, checking NSPs synchronization.--')
            check_parser_sync([NSx(sinc_nsp1).output_name NSx(sinc_nsp1).ext],[NSx(sinc_nsp2).output_name NSx(sinc_nsp2).ext]);
        end
    end
end
end


function unit = max_analog2unit(x)
    switch x
        case 5000
            unit='mV';
        case 8191
            unit ='uV';
    end
end

function [init_cells, maxlts, trims4sinc] = select_init_cells(filenames, which_nsp)
    assert(numel(which_nsp)==numel(filenames),'filenames and which_nsp should have the same number of elements')
    assert(sum(which_nsp==1)==sum(which_nsp==2),'Each which_nsp should have the same number of files')

    filepos = @(i) sum(which_nsp(1:i)==which_nsp(i));%given the file index get the position in series of that nsp

    lts_mat = zeros(2,sum(which_nsp==1));%first row nsp 1, second row nsp 2
    init_cells = zeros(2,sum(which_nsp==1));
    maxlts = zeros(2,sum(which_nsp==1));
    for fi = 1:length(filenames)
        nsx_file = openNSx(filenames{fi}, 'report','noread');
        DataPoints = nsx_file.MetaTags.DataPoints;

        if length(DataPoints)>1
            fprintf('\n\n')
            fprintf('###################\n')
            fprintf('Filename: %s\n', filenames{fi})
            fprintf('NSx.MetaTags.Timestamp: %s', formatvector(nsx_file.MetaTags.Timestamp,'.f'))
            fprintf('NSx.MetaTags.DataPoints: %s', formatvector(nsx_file.MetaTags.DataPoints,'.f'))
            fprintf('NSx.MetaTags.DataDurationSec: %s', formatvector(nsx_file.MetaTags.DataDurationSec,'.5f'))
            init_cell = [];
            while isempty(init_cell) || (init_cell> length(DataPoints)) || (init_cell<1)
                if ~isempty(init_cell)
                    warning('wrong answer')
                end
                init_cell = input(sprintf('From which cell number you want to save data? From 1 to %d: ',length(DataPoints)));
            end
        else
            init_cell = 1;
        end

        %total lenght adding the zeros from Timestamp
        fp = filepos(fi);
        init_cells(which_nsp(fi),fp) = init_cell;
        Timestamp = round(nsx_file.MetaTags.Timestamp(init_cell:end)*nsx_file.MetaTags.SamplingFreq/nsx_file.MetaTags.TimeRes);
        cellslts = zeros(numel(nsx_file.MetaTags.Timestamp(init_cell:end)),1);
        cellslts(1) = floor(nsx_file.MetaTags.DataPoints(init_cell)) + Timestamp(1);
        
        for ci=2:numel(cellslts)
            cellslts(ci)=Timestamp(ci)-sum(cellslts(1:ci-1))+ floor(nsx_file.MetaTags.DataPoints(init_cell+ci-1));
        end
        lts_mat(which_nsp(fi),fp)=sum(cellslts);
    end
    
    lts_min = min(lts_mat,[],1); %get the min for each
    trims4sinc = lts_mat-lts_min;
    for fi = 1:length(filenames)
        fp = filepos(fi);
        maxlts(which_nsp(fi),fp) = lts_min(fp);
    end
end


function fv = formatvector(v,f)
    fv=sprintf(['[' repmat(['%' f ', '], 1, numel(v)-1), ['%' f ']\n']  ],v);
end