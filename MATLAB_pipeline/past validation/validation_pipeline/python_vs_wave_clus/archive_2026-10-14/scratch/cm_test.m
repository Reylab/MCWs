P = '/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline';
addpath(fullfile(P,'validation_pipeline'), fullfile(P,'real_response_analysis'), fullfile(P,'responses'));
root = '/mnt/data0/MCWs Sessions/matlab validation/single channel/pt 11 emu 17 ch 328';
ch = 328; gt_class = 1; f = 'times_mLFUS08 raw_328.mat';
gt  = load(fullfile(root, 'hand selected', f));
arm = load(fullfile(root, 'wave_clus_20261008', f));
gt_t = sort(gt.cluster_class(gt.cluster_class(:,1) == gt_class, 2));
[cls, ~, DET, COMP] = match_arms_to_ground_truth([repmat(gt_class, numel(gt_t), 1) gt_t], gt_t, ...
    {'wave_clus'}, {arm.cluster_class}, ch, 0.5, 0.5);
out = fullfile(pwd, 'cm_out'); if ~exist(out, 'dir'), mkdir(out); end
class_match_figure(DET, COMP, ch, {'wave_clus'}, out);
fprintf('matched class %d\n', cls); disp(ls(out))
