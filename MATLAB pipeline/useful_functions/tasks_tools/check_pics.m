function check_pics(folder)
%this code check the format, dimension and name of the pictures in folder
%and subfolders
if ~exist('folder','var')
    folder = pwd;
end
check_amount_per_identity = true; %if true it will check the following number of pictures per concept
expected_per_identity = 3;
rename2lower = false;
if rename2lower && ispc
    rename2lower = false;
    warning('Windows is not case sensitive')
end
test_number = true;

files = dir([folder '/**/*']);

wrong_files = 0;
for i =1:length(files)
    iswrong = 0;
    if files(i).isdir
        continue
    end
    pic_path = fullfile(files(i).folder,files(i).name);
    if rename2lower
        if ~strcmp(files(i).name,lower(files(i).name))
            new_name = lower(files(i).name);
            new_pic_path = fullfile(files(i).folder,new_name);
            movefile(pic_path, new_pic_path)
            files(i).name = new_name;
            pic_path = new_pic_path;
        end
    end
    if test_number
        concept_number = str2double(cell2mat(regexpi(files(i).name,'_(\d*).\D*$','tokens','once')));
        if isnan(concept_number)
            warning('Missing _# in filename in %s',pic_path)
            iswrong= 1;
        end
    end
    [~,~,ext]=fileparts(files(i).name);
    if length(ext)<4 || ~strcmpi(ext(2:3),'jp')
        warning('File without .jp* extension: %s',pic_path)
        iswrong= 1;
    end
    if ~strcmp(strtrim(files(i).name),files(i).name)
        warning('Leading or ending whitespaces in %s',pic_path)
        iswrong= 1;
    end
    Im = imread(pic_path);
    nRows=size(Im,1); nCols=size(Im,2);
    if nRows~=160 || nCols~=160
        warning('Wrong dimentions: %s', pic_path)
        iswrong= 1;
    end
    wrong_files = wrong_files + iswrong;
end

if wrong_files>0
    warning('%d Files with naming issues',wrong_files)
else
    disp('All the pictures names are fine.')
end


a = files;
filt_a = arrayfun(@(x) contains (lower(x.name), '.jp'),a);
a = a(filt_a);
ImageNames = array2table({a.name}','VariableNames',{'name'});

ImageNames.identity = cellfun(@(x) regexp(x,'^.*(?=(_\d*(?s)\D*$))','match','once'),ImageNames.name,'UniformOutput',false);

without_numbers = cellfun('isempty',ImageNames.identity); %this could be done with a regular expression
ImageNames.identity(without_numbers) = cellfun(@(x) regexp(x,'^.*(?=((?s)\D*$))','match','once'),ImageNames.name(without_numbers),'UniformOutput',false);

ImageNames.concept_name = cellfun(@(x) regexp(x,'[^~]*$','match','once'),ImageNames.identity,'UniformOutput',false);

if check_amount_per_identity
    [u,~,IC] = unique(ImageNames.identity,'stable');
    for iu = 1: numel(u)
        if  expected_per_identity ~= sum(IC==iu)
           warning('%d pictures found with the identity %s', sum(IC==iu), u{iu})
        end
    end
end
fprintf('--Stats--\ntotal pictures: %d\n\n', length(ImageNames.concept_name))
[u,~,IC] = unique(ImageNames.concept_name,'stable');
concept_counter = histcounts(IC,1: numel(u)); %number of pictures for each concept
concept_hist = histcounts(concept_counter,1: max(concept_counter+1)); %histogram of number of pictures for each concept
for iu = 1: numel(concept_hist)
    if concept_hist(iu)>0
       fprintf('%d concepts found with %d associated pictures.\n', concept_hist(iu), iu)
    end
end
fprintf('total concepts: %d\n\n', sum(concept_hist))

