function cfg = validation_cfg_pt10_emu7_ch368_norefract()
% VALIDATION_CFG_PT10_EMU7_CH368_NOREFRACT  Same session/GT as
% validation_cfg_pt10_emu7_ch368, but the arms are the Python reruns whose
% rescue excludes refractory-flagged spikes (rescue_refractory=False), at
% sd1 and sd2, on the same clustering.

cfg = validation_cfg_pt10_emu7_ch368();

cfg.arms = struct('name',        {'python_norefract_sd1', 'python_norefract_sd2'}, ...
                  'dir',         {fullfile(cfg.root, 'python_rescue_no_refract_sd1'), ...
                                  fullfile(cfg.root, 'python_rescue_no_refract_sd2')}, ...
                  'times_file',  {'times_mRFUS07 raw_368.mat', 'times_mRFUS07 raw_368.mat'}, ...
                  'grapes_file', {'grapes_blanks_rescue.mat', 'grapes_blanks_rescue.mat'}, ...
                  'class',       {[], []});

cfg.out_dir = fullfile(cfg.root, 'test_resp_gt_vs_python_sd1_sd2');
end
