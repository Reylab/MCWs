P = '/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline';
addpath(genpath(P));
V = fullfile(P, 'validation_pipeline');
entry = {'run_validation', 'rerun_validation', 'plot_subset_rasters', 'check_validation_run', 'export_validation_results', ...
         'make_validation_report', 'convert_recluster_runs', 'build_wave_clus_arm', 'save_validation_subsets', 'load_validation_state', ...
         'build_python_arm_from_times'};
cfgs = dir(fullfile(V, 'validation_cfg_*.m'));
files = [cellfun(@(e) which(e), entry, 'UniformOutput', false), fullfile(V, {cfgs.name})];
req = matlab.codetools.requiredFilesAndProducts(files);
req = req(:);
fid = fopen(fullfile(fileparts(mfilename('fullpath')), 'deps.txt'), 'w');
fprintf(fid, '%s\n', req{:}); fclose(fid);
fprintf('%d required files\n', numel(req));
