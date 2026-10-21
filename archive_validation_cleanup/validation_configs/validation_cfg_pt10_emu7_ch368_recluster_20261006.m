function cfg = validation_cfg_pt10_emu7_ch368_recluster_20261006()
% VALIDATION_CFG_PT10_EMU7_CH368_RECLUSTER_20261006  Same session, GT (hand
% selected, class 3) and stim list as validation_cfg_pt10_emu7_ch368, with a
% single arm: the Python recluster run
%   /mnt/data0/python_p10EMU07CH368/times_2026-10-06_13-46-07
% made on another machine. What it did, from its own par.second_pass:
%   'split: c3; templates from SPC cores; re-match std1->3; rescue artifacts
%    std1->1; amp window off; Mahalanobis inside destination radius'
% i.e. the rescue was sd1 (its par.template_sdnum = 2 / amp_pct_range [1 99]
% are leftover fields, not what ran). Converted to python_recluster_20261006_sd1/
% with build_python_arm_from_times.

cfg = validation_cfg_pt10_emu7_ch368();

cfg.arms = struct('name',        {'python_recluster_sd1'}, ...
                  'dir',         {fullfile(cfg.root, 'python_recluster_20261006_sd1')}, ...
                  'times_file',  {'times_mRFUS07 raw_368.mat'}, ...
                  'grapes_file', {'grapes_blanks_rescue.mat'}, ...
                  'class',       {[]});

cfg.out_dir = fullfile(cfg.root, 'validation_gt_vs_python_recluster_20261006_sd1');
cfg.venn_detections = true;   % also the all-detections venn (GT vs arm detectors)
cfg.channel_subsets = true;   % also save full-venn and detections-venn subsets (whole channel)
end
