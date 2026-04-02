stim_num = 1;

exp_props_file = 'experiment_properties_online3.mat';
load(exp_props_file);

img_path = fullfile(experiment.params.pics_root_processing,experiment.ImageNames.folder{stim_num}, ...
                    experiment.ImageNames.name{stim_num});

temp_folder = '/mnt/acq-hdd/temp/';
if ~exist(temp_folder, 'dir')
   mkdir(temp_folder)
end

miniscr_folder = [experiment.params.pics_root_processing filesep 'miniscr_pics'];

copy_file(img_path, miniscr_folder, temp_folder);
