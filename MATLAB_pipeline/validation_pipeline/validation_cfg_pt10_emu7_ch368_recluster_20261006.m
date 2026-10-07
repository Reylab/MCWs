function cfg = validation_cfg_pt10_emu7_ch368_recluster_20261006()
% VALIDATION_CFG_PT10_EMU7_CH368_RECLUSTER_20261006  Same session, GT (hand
% selected, class 3) and stim list as validation_cfg_pt10_emu7_ch368, with a
% single arm: the Python recluster run
%   /mnt/data0/python_p10EMU07CH368/times_2026-10-06_13-46-07
% (rescue sd1 -> sd2 -> Mahalanobis, amplitude gate [1 99], within-channel
% min_amplitude_percentile 5%; saved before rescue_refractory existed, so it
% DID rescue refractory-flagged spikes), converted to
% python_recluster_20261006_sd2/ with build_python_arm_from_times.

cfg = validation_cfg_pt10_emu7_ch368();

cfg.arms = struct('name',        {'python_recluster_sd2'}, ...
                  'dir',         {fullfile(cfg.root, 'python_recluster_20261006_sd2')}, ...
                  'times_file',  {'times_mRFUS07 raw_368.mat'}, ...
                  'grapes_file', {'grapes_blanks_rescue.mat'}, ...
                  'class',       {[]});

cfg.out_dir = fullfile(cfg.root, 'validation_gt_vs_python_recluster_20261006_sd2');
end
