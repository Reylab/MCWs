
channels = [257:264 266:273];

channels = [263];

nfigures = 2;
output_folder = ''; 
times_folder = 'D:\fertest\EMU-005_task-RSVPscr_dynamic_run-03\results\new_with_wc';
online_data_folder = 'D:\fertest\EMU-005_task-RSVPscr_dynamic_run-03\results\with_wc';
path_pics = 'D:\fertest\EMU-005_task-RSVPscr_dynamic_run-03\all_pics';
load('D:\fertest\EMU-005_task-RSVPscr_dynamic_run-03\experiment_properties_online3.mat','experiment','scr_config_cell')
experiment.ImageNames.folder(:)={''};
cd(output_folder)
make_wc_rasters_online(channels,nfigures,experiment,scr_config_cell,times_folder,online_data_folder,path_pics)