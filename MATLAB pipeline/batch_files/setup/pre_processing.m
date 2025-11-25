function varargout = pre_processing(x,id)
if ~exist('pre_processing_info.mat','file')
    error('pre_processing_info.mat not found')
end
load('pre_processing_info.mat','process_info')

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