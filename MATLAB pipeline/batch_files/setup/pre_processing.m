function varargout = pre_processing(x,id)

potential_paths = {
    fullfile(pwd, 'preprocessing', 'pre_processing_info.mat'), ...
    fullfile(pwd, 'preprocessing info', 'pre_processing_info.mat'), ...
    fullfile(pwd, 'preprocessing_info', 'pre_processing_info.mat'), ...
    fullfile(pwd, 'pre_processing_info.mat')
};

file_preprocess = '';
for i = 1:length(potential_paths)
    if exist(potential_paths{i}, 'file')
        file_preprocess = potential_paths{i};
        break;
    end
end

if isempty(file_preprocess)
    error('pre_processing_info.mat not found in any of the expected locations.')
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