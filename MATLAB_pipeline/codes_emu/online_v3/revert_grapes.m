function new_grapes = revert_grapes(grapes,subscr,scr_config_cell)
new_grapes = grapes;
npics = size(grapes.ImageNames,1);
pic_counter =  zeros(npics,1);

for ss = 1:subscr
    picsused = scr_config_cell{ss}.pics2use;
    pic_counter(picsused)= pic_counter(picsused)+scr_config_cell{ss}.Nrep;
end
channels = fieldnames(new_grapes.rasters);
for ci = 1:numel(channels)
    units = fieldnames(new_grapes.rasters.(channels{ci}));
    for ui=1:numel(units)
        if strcmp(units{ui},'details')
            continue
        end
        for i= 1:npics
            if numel(new_grapes.rasters.(channels{ci}).(units{ui}).stim)<i
                continue
            end
            new_grapes.rasters.(channels{ci}).(units{ui}).stim{i} = new_grapes.rasters.(channels{ci}).(units{ui}).stim{i}(1:pic_counter(i));
        end
    end
end

end