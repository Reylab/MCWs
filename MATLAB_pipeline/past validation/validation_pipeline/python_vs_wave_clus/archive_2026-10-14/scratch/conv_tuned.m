P = '/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline';
addpath(fullfile(P,'validation_pipeline'), fullfile(P,'real_response_analysis'));
addpath(genpath(fullfile(P,'codes_emu'))); addpath(genpath(fullfile(P,'useful_functions')));
C = jsondecode(fileread(fullfile(P, 'validation_pipeline/python_vs_wave_clus/channels.json')));
done = [];
for u = C.units'
    if ismember(u.ch, done), continue; end
    done(end+1) = u.ch; %#ok<SAGROW>
    root = fullfile(C.single_channel, u.folder);
    f = dir(fullfile(C.recluster_runs, u.run, 'tuned_width_logdet', 'second_pass_width_logdet', '*', 'times_*', 'times_*.mat'));
    g = dir(fullfile(root, 'hand selected', 'grapes*.mat')); x = load(fullfile(g(1).folder, g(1).name), 'exp_type');
    build_python_arm_from_times('root', root, 'channel', u.ch, 'exp_type', x.exp_type, ...
        'arm_name', 'python_tuned_width_logdet', 'python_times_file', fullfile(f(1).folder, f(1).name));
    d = fullfile(root, 'python_tuned_width_logdet');
    T = load(fullfile(d, u.times_file), 'cluster_class'); lab = T.cluster_class(:,1);
    tc = arrayfun(@(c) nnz(lab==c), 1:max(lab));
    G = load(fullfile(d, 'grapes_blanks_rescue.mat'), 'rasters'); dt = G.rasters.(sprintf('chan%d', u.ch)).details;
    gc = arrayfun(@(c) dt.(sprintf('class%d', c)), 1:numel(tc));
    fprintf('[conv] ch%d times %s grapes %s match %d\n', u.ch, mat2str(tc), mat2str(gc), isequal(tc, gc));
end
