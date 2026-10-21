function resp_build(folder, ch, run, wc_run)
% Response metrics for the KS / GMM test sorts, wave_clus and GT, built in a
% scratch session root (copies of the channel folder's session files; the
% channel folder itself is only read). Per sort: grapes (same recipe as the
% validation arms: build_python_arm_from_times) then characterize_grapes_responses
% -> <scratch>/<ch>/<sort>/grapes_response_metrics_ch<ch>.mat
P = '/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline';
addpath(fullfile(P, 'responses'), fullfile(P, 'real_response_analysis'), fullfile(P, 'validation_pipeline'));
addpath(genpath(fullfile(P, 'codes_emu')), genpath(fullfile(P, 'useful_functions')));
SC = '/mnt/data0/MCWs Sessions/matlab validation/single channel';
src = fullfile(SC, folder);
root = fullfile('/tmp/claude-1017/-home-ajb-Documents-GitHub-MCWs-MATLAB-pipeline/1d8fd482-04b6-49a4-9526-6897f90343d6/scratchpad/gmm_test/resp', sprintf('ch%d', ch));
if ~exist(root, 'dir'), mkdir(root); end
for f = {'NSx.mat', 'NSX_TimeStamps.mat', 'stimulus.mat', 'finalevents.mat', 'experiment_properties_online3.mat', 'pre_processing_info.mat'}
    if exist(fullfile(src, f{1}), 'file') && ~exist(fullfile(root, f{1}), 'file'), copyfile(fullfile(src, f{1}), root); end
end
if ~exist(fullfile(root, 'pics_used'), 'dir'), system(sprintf('ln -s "%s" "%s"', fullfile(src, 'pics_used'), fullfile(root, 'pics_used'))); end
g = dir(fullfile(src, 'hand selected', 'grapes*.mat'));
exp_type = getfield(load(fullfile(g(1).folder, g(1).name), 'exp_type'), 'exp_type'); %#ok<GFLD>
R = fullfile('/mnt/data0/recluster_runs', run, 'gmm_feature_test');
sorts = {};
for v = {'ks', 'gmm_seed0', 'gmm_seed1'}
    t = dir(fullfile(R, v{1}, 'second_pass_*', '*', 'times_*', 'times_*.mat'));
    sorts(end+1, :) = {['python_' v{1}], fullfile(t(end).folder, t(end).name)}; %#ok<AGROW>
end
old = pwd;
for s = 1:size(sorts, 1)
    D = fullfile(root, sorts{s, 1});
    if ~exist(fullfile(D, 'grapes_blanks_rescue.mat'), 'file')
        build_python_arm_from_times('root', root, 'channel', ch, 'exp_type', exp_type, 'arm_name', sorts{s, 1}, 'python_times_file', sorts{s, 2});
    end
    cd(D); characterize_grapes_responses('channels2plot', ch, 'session_root', root, 'grapes_files', {'grapes_blanks_rescue.mat'});
end
% wave_clus and GT: link their grapes (read only) into scratch folders and score there
ref = {{'GT', fullfile(src, 'hand selected')}};
if ~isempty(wc_run), ref = [{{'wave_clus', fullfile(src, wc_run)}}, ref]; end
for s = ref
    D = fullfile(root, s{1}{1}); if ~exist(D, 'dir'), mkdir(D); end
    gf = dir(fullfile(s{1}{2}, 'grapes_blanks.mat'));
    assert(~isempty(gf), 'no grapes_blanks.mat in %s', s{1}{2});
    if ~exist(fullfile(D, 'grapes_blanks.mat'), 'file'), system(sprintf('ln -s "%s" "%s"', fullfile(gf.folder, gf.name), fullfile(D, 'grapes_blanks.mat'))); end
    cd(D); characterize_grapes_responses('channels2plot', ch, 'session_root', root, 'grapes_files', {'grapes_blanks.mat'});
end
cd(old);
fprintf('resp_build ch%d done -> %s\n', ch, root);
end
