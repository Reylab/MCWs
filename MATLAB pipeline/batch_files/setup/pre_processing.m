function varargout = pre_processing(x,id)

file_preprocess = fullfile(pwd, 'preprocessing', 'pre_processing_info.mat');
if ~exist(file_preprocess,'file')
    if ~exist('pre_processing_info.mat', 'file')
        error('pre_processing_info.mat not found')
    else
        file_preprocess = 'pre_processing_info.mat'; % Fallback to current directory for old data
    end
end
load(file_preprocess,'process_info')

if ischar(id)
    f = regexp(regexp(id,'_\d+(.|$)','match','once'),'\d+','match','once'); %it can parse filename to id
    if isempty(f)
        error('id not foun in filename.')
    end
    id = str2num(f);
end


%check fields names
if ~isempty(process_info)
    index = find([process_info(:).chID]==id);
else
    index = [];
end
if ~isempty(index) && ~isempty(x)
    x = fast_filtfilt(process_info(index).SOS,process_info(index).G,x);
end

varargout{1} = x;
if (nargin == 2)
    varargout{2} = process_info(index);
end
end