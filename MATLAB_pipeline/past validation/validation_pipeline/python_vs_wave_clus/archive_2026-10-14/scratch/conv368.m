P = '/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline';
addpath(fullfile(P,'validation_pipeline'), fullfile(P,'real_response_analysis'));
addpath(genpath(fullfile(P,'codes_emu'))); addpath(genpath(fullfile(P,'useful_functions')));
root = '/mnt/data0/MCWs Sessions/matlab validation/single channel/pt 10 emu 7 ch 368';
g = dir(fullfile(root, 'hand selected', 'grapes*.mat'));
x = load(fullfile(g(1).folder, g(1).name), 'exp_type');
build_python_arm_from_times('root', root, 'channel', 368, 'exp_type', x.exp_type, ...
    'arm_name', 'python_recluster_20261013_second_pass', 'python_times_file', '/mnt/data0/recluster_runs/p10_EMU-007_ch368/10_12_reclust_test/second_pass_2026-10-13_12-25-41/mRFUS07_raw_368/times_2026-10-13_12-26-33/times_mRFUS07 raw_368.mat');
d = fullfile(root, 'python_recluster_20261013_second_pass');
T = load(fullfile(d, 'times_mRFUS07 raw_368.mat'), 'cluster_class'); lab = T.cluster_class(:,1);
tc = arrayfun(@(c) nnz(lab==c), 1:max(lab));
for f = {'grapes_blanks.mat','grapes_blanks_rescue.mat'}
    G = load(fullfile(d, f{1}), 'rasters'); dt = G.rasters.chan368.details;
    gc = arrayfun(@(c) dt.(sprintf('class%d', c)), 1:numel(tc));
    fprintf('[conv] %s times %s grapes %s %s\n', f{1}, mat2str(tc), mat2str(gc), string(isequal(tc,gc)));
end
build_python_arm_from_times('root', root, 'channel', 368, 'exp_type', x.exp_type, ...
    'arm_name', 'python_20261001_rescued', 'python_times_file', '/mnt/data0/python_p10EMU07CH368/Rescue_STD2/times_2026-10-01_16-23-02/times_mRFUS07 raw_368.mat');
T = load(fullfile(root, 'python_20261001_rescued', 'times_mRFUS07 raw_368.mat'), 'cluster_class'); lab = T.cluster_class(:,1);
fprintf('[conv] before-second-pass arm classes %s\n', mat2str(arrayfun(@(c) nnz(lab==c), 1:max(lab))));
