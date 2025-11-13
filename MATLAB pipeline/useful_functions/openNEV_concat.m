function NEV = openNEV_concat(varargin)
%openNEV_concat concatenate nev files after using parse_data_NSx with the
%ns5 files.
%   Inputs: cell of filenames or a filename as each input [in order].
%   Ouput: NEV struct with the concatenated and realing nev data.

if length(varargin)==1
    if ischar(varargin{1}) %just one string
       inputs =  varargin(1);
    else %one cell
       inputs =  varargin{1};
    end
else
    inputs = varargin;
end

load('Nsx.mat','files');

file_start = zeros(length(inputs),1);
for i = 1:length(inputs)
    if ~exist(inputs{i},'file')
        error('ERROR: File %s doesn''t exist.',inputs{i})
    end
    
    db_ix = logical(cellfun(@(x) ns5_match(x,inputs{i}),{files.name}));
    
    if ~any(db_ix)
        error('ERROR: File %s doesn''t have an associated ns5 file in Nsx.mat.',inputs{i})
    elseif sum(db_ix)>1
        error('ERROR: File %s has multiple associated ns5 files in Nsx.mat.',inputs{i})
    end
    file_start(i) = files(db_ix).first_sample-1;
    
end

[~,argsort] = sort(file_start,'ascend');

inputs = inputs(argsort);
file_start = file_start(argsort);
NEV = openNEV(file2path(inputs{1}),'report','noread','8bits');
NEV.MetaTags.Filename = inputs;
file_sr = double(NEV.MetaTags.DataDuration)/NEV.MetaTags.DataDurationSec;

%move times for first file (typically will will be used)
if ~isempty(NEV.Data.SerialDigitalIO.UnparsedData)
    NEV.Data.SerialDigitalIO.TimeStamp = NEV.Data.SerialDigitalIO.TimeStamp + file_start(1);
    NEV.Data.SerialDigitalIO.TimeStampSec = NEV.Data.SerialDigitalIO.TimeStampSec +  double(file_start(1))/file_sr;
end

for i = 2:length(inputs)
    file = inputs{i};
    next_nev = openNEV(file2path(file),'report','noread','8bits');
    
    %update SerialDigitalIO
    if ~isempty(next_nev.Data.SerialDigitalIO.UnparsedData)
        NEV.Data.SerialDigitalIO.UnparsedData = [NEV.Data.SerialDigitalIO.UnparsedData next_nev.Data.SerialDigitalIO.UnparsedData];
        NEV.Data.SerialDigitalIO.InsertionReason = [NEV.Data.SerialDigitalIO.InsertionReason next_nev.Data.SerialDigitalIO.InsertionReason];
        
        
        NEV.Data.SerialDigitalIO.TimeStamp = [NEV.Data.SerialDigitalIO.TimeStamp,...
            (next_nev.Data.SerialDigitalIO.TimeStamp  + file_start(i))];
        NEV.Data.SerialDigitalIO.TimeStampSec = [NEV.Data.SerialDigitalIO.TimeStampSec,...
            (next_nev.Data.SerialDigitalIO.TimeStampSec+  file_start(i)/file_sr)];
    end
    
    %update comments
    if ~isempty(next_nev.Data.Comments.TimeStamp)
        NEV.Data.Comments.Color = [NEV.Data.Comments.Color ; next_nev.Data.Comments.Color];
        NEV.Data.Comments.Text = [NEV.Data.Comments.Text ; next_nev.Data.Comments.Text];
        NEV.Data.Comments.CharSet = [NEV.Data.Comments.CharSet , next_nev.Data.Comments.CharSet];

        NEV.Data.Comments.TimeStamp = [NEV.Data.Comments.TimeStamp,...
            (next_nev.Data.Comments.TimeStamp  + file_start(i))];
        NEV.Data.Comments.TimeStampSec = [NEV.Data.Comments.TimeStampSec,...
            (next_nev.Data.Comments.TimeStampSec+  file_start(i)/file_sr)];
    
        NEV.Data.Comments.TimeStampStarted = [NEV.Data.Comments.TimeStampStarted,...
            (next_nev.Data.Comments.TimeStamp  + file_start(i))];
        
        NEV.Data.Comments.TimeStampStartedSec = [NEV.Data.Comments.TimeStampStartedSec,...
            (next_nev.Data.Comments.TimeStampSec+  file_start(i)/file_sr)];
    
    end    
    
    %update MetaTags
    NEV.MetaTags.DataDuration = NEV.MetaTags.DataDuration + next_nev.MetaTags.DataDuration;
    NEV.MetaTags.DataDurationSec = NEV.MetaTags.DataDurationSec + next_nev.MetaTags.DataDurationSec;
    NEV.MetaTags.PacketCount = NEV.MetaTags.PacketCount + next_nev.MetaTags.PacketCount;
end

end

function newf = file2path(file)
    if file(1) ~= '.'
        newf = ['.' filesep file];
    else
        newf = file;
    end
end

function value = ns5_match(full_filename,input)
    [~,name,ext] = fileparts(full_filename);
    value = false;
    if strcmp(ext,'.ns5')
        [~,nameinput,~] = fileparts(input);
        value = strcmp(name,nameinput);
    end
end