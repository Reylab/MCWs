function ImageNames = load_dynamic_scr_img_info(ImageNames)
ImageNames.concept_name = cellfun(@(x) regexpi(x,'^.*(?=(_\d*(?s)\D*$))','match','once'),ImageNames.name,'UniformOutput',false);

without_numbers = cellfun('isempty',ImageNames.concept_name); %this could be done with a regular expression
ImageNames.concept_name(without_numbers) = cellfun(@(x) regexpi(x,'^.*(?=((?s)\D*$))','match','once'),ImageNames.name(without_numbers),'UniformOutput',false);


ImageNames.concept_number = cellfun(@(x)str2double(cell2mat(regexpi(x,'_(\d*).\D*$','tokens','once'))),ImageNames.name);
ImageNames.concept_number(isnan(ImageNames.concept_number))=1;

[u,~,IC] = unique(ImageNames.concept_name,'stable');

for iu = 1: numel(u)
    ImageNames.concept_number(IC==iu) = 1:sum(IC==iu);
end


ImageNames = sortrows(ImageNames,'concept_number');


ImageNames.concept_categories = cellfun(@(x)strsplit(x,'~'),ImageNames.concept_name,'UniformOutput',false);

%removing categories:
categories2remove = {'animal','hobby'};

for i = 1:height(ImageNames)
    ctgs = ImageNames.concept_categories{i};
    valid_ctgs = cellfun(@(x) ~any(strcmp(x, categories2remove)),ctgs);
    ImageNames.concept_categories{i} = ctgs(valid_ctgs);
end

end