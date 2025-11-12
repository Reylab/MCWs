function parse_data_NSx(filenames,which_nsp ,max_memo_GB ,output_name,channels)
% Code only valid for recordings without pauses or data loss (Nsx.Data field can't be a cell)
% This code requires the file openNSx.m, from the NPMK, in the path. You can download the download NPMK from https://github.com/BlackrockMicrosystems/NPMK/releases
% max_memo_GB is an idea of the number of GB allocated for the data to be
% stored in RAM, so it is used to compute the number of segments in which
% the data should be split for processing
if ~exist('output_name','var') || isempty(output_name)
    output_name = 'NSX';
end
if ~exist('which_nsp','var')
    which_nsp = [];
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
	max_memo = max_memo_GB*(1024)^3;
end

tcum=0;
if ischar(filenames)
    filenames = {filenames};
end

formatvector=@(v) sprintf(['[' repmat('%g, ', 1, numel(v)-1), '%g]\n'  ],v);

for fi = 1:length(filenames)
    filename= filenames{fi};
    new_files(fi).name = filename;
    if length(filename)<3 || (~strcmpi(filename(2:3),':\') && ...
                     ~strcmpi(filename(1),'/') && ...
                     ~strcmpi(filename(2),'/') && ...
                     ~strcmpi(filename(1:2), '\\')&& ~strcmpi(filename(2:3),':/'))

        filename= [pwd filesep filename];
    end
    
    NSx = openNSx(filename, 'report','noread');
    if isnumeric(NSx) && NSx==-1
        error('Unable to open file: %s',filename)
    end
    nchan = NSx.MetaTags.ChannelCount;   % number of channels
    if fi == 1
        outfile_handles = cell(1,nchan); %some will be empty
        [~,~,fext] = fileparts(filename);
        fext = lower(fext(2:end));
        nsx_ext = fext(end);
        ch_ext = ['.NC' nsx_ext];
        if ~exist('channels','var') || isempty(channels)
            channels = NSx.MetaTags.ChannelID;
        end
        parsed_chs = [];
        new_channel_id = [];
        for i = 1:nchan
            c = NSx.MetaTags.ChannelID(i);
            if ismember(c,channels)
                parsed_chs(end+1) = c;
                ccname = c;
                if ~isempty(which_nsp)
                    ccname = ccname + 1000*which_nsp;
                end
                new_channel_id(end+1) = ccname;
                outfile_handles{i} = fopen([output_name '_' num2str(ccname) ch_ext],'w');
            end
        end
        
        chs_info = struct();
        chs_info.unit = cellfun(@(x) max_analog2unit(x),{NSx.ElectrodesInfo.MaxAnalogValue},'UniformOutput',false)';
        chs_info.label = cellfun(@(x) deblank(x),{NSx.ElectrodesInfo.Label},'UniformOutput',false)';
        chs_info.conversion = (double(cell2mat({NSx.ElectrodesInfo.MaxAnalogValue}))./double(cell2mat({NSx.ElectrodesInfo.MaxDigiValue})))';
        chs_info.id = cell2mat({NSx.ElectrodesInfo.ElectrodeID});
        
        new_files(fi).first_sample = 1;
    else
        new_files(fi).first_sample = new_files(fi-1).lts + new_files(fi-1).first_sample;
    end

    DataPoints = NSx.MetaTags.DataPoints;
    
    if length(DataPoints)>1 && ~isempty(which_nsp)
        fprintf('\n')
        fprintf('###################\n')
        fprintf('NSx.MetaTags.Timestamp: %s', formatvector(NSx.MetaTags.Timestamp))
        fprintf('NSx.MetaTags.DataPoints: %s', formatvector(NSx.MetaTags.DataPoints))
        fprintf('NSx.MetaTags.DataDurationSec: %s', formatvector(NSx.MetaTags.DataDurationSec))
        init_cell = [];
        while isempty(init_cell) || (init_cell> length(DataPoints)) || (init_cell<1)
            if ~isempty(init_cell)
                warning('wrong answer')
            end
            init_cell = input(sprintf('From which cell number yo want to save data? From 1 to %d:',length(DataPoints)));
        end
    else
        init_cell = 1;
    end
    
    sr = NSx.MetaTags.SamplingFreq;   % sampling rate
    %total lenght adding the zeros from Timestamp
    lts = floor(sum(NSx.MetaTags.DataPoints(init_cell:end))) + sum(round(NSx.MetaTags.Timestamp(init_cell:end)*NSx.MetaTags.SamplingFreq/NSx.MetaTags.TimeRes));
    
    
    new_files(fi).lts = lts;
    new_files(fi).which_cells = init_cell:length(DataPoints);
    
    DataPoints = [0 DataPoints]; %added for use as starting index
    samples_per_channel = ceil(max_memo/(nchan*length(NSx.MetaTags.DataPoints))/2);
    for part = 1 + init_cell:length(DataPoints)
        N = floor(DataPoints(part));   % total data points 
        num_segments = ceil(N/samples_per_channel);
        fprintf('Data will be processed in %d segments of %d samples each.\n',num_segments,min(samples_per_channel,N))
        for j=1:num_segments
            ini = (j-1)*samples_per_channel+1+floor(DataPoints(part-1));
            fin = min(j*samples_per_channel,N)+floor(DataPoints(part-1));
            tcum = tcum + toc;  % this is because openNSx has a tic at the beginning
            NSx = openNSx('read',filename,['t:' num2str(ini) ':' num2str(fin)]);
            zeros2add = round(NSx.MetaTags.Timestamp*NSx.MetaTags.SamplingFreq/NSx.MetaTags.TimeRes);
            for i = 1:nchan
                if ~isempty(outfile_handles{i}) %channels with empty outfile_handles{i} are not selected
                    if (j==1) && (NSx.MetaTags.Timestamp>0)
                        fwrite(outfile_handles{i},zeros(zeros2add,1,'int16'),'int16');
                    end
                    fwrite(outfile_handles{i},NSx.Data(i,1:end),'int16');
                end
            end
            fprintf('Segment %d out of %d processed.',j,num_segments)
            if j==1
                fprintf('Zeros added = %d. ',zeros2add);
            end
            fprintf('Data Point Read = %d \n',size(NSx.Data,2));
        end
    end
    
    tcum = tcum + toc;
    fprintf('Total time spent in parsing the data was %s secs.\n',num2str(tcum, '%0.1f')); 
    if length(DataPoints)>2 && isempty(which_nsp)
        warning('Automatically merged %d cells of data.', length(DataPoints)-1 )
        fprintf('NSx.MetaTags.Timestamp: %s', formatvector(NSx.MetaTags.Timestamp))
        fprintf('NSx.MetaTags.DataPoints: %s', formatvector(NSx.MetaTags.DataPoints))
    end
end
fclose('all');



metadata_file = fullfile(pwd, 'NSx.mat');
if exist(metadata_file,'file')
    metadata = load(metadata_file);
    NSx = metadata.NSx;
    files = metadata.files;
else
    files = [];
    NSx = [];
end
lts = sum([new_files(:).lts]);
fprintf('%d data points written per channel\n',lts)


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
        NSx(pos).nsp = which_nsp;
        NSx(pos).ext = ch_ext;
        NSx(pos).lts = lts;
        NSx(pos).filename = filenames;
        NSx(pos).sr = sr;
        NSx(pos).output_name = output_name;
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
    files(pos).which_nsp = which_nsp;
    files(pos).which_cells = init_cell:(length(DataPoints)-1);
end

save(metadata_file, 'NSx','files')
end


function unit = max_analog2unit(x)
    switch x
        case 5000
            unit='mV';
        case 8191
            unit ='uV';
    end
end
