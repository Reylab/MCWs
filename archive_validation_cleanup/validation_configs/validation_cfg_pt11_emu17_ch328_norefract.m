function cfg = validation_cfg_pt11_emu17_ch328_norefract()
% VALIDATION_CFG_PT11_EMU17_CH328_NOREFRACT  Config for run_validation:
% pt 11 / EMU 17 / ch328, hand-selected GT (class 1) vs the Python rescue
% reruns that exclude refractory-flagged spikes (rescue_refractory=False), at
% sd1 and sd2. Same pre-rescue clustering as python_rescue_test /
% python_rescue_test_sd2 (verified identical labels and spike times).

cfg.root = '/mnt/data0/MCWs Sessions/matlab validation/single channel/pt 11 emu 17 ch 328';
cfg.ch = 328;
cfg.stim_list = [76 386 858 69 391 875 139 1015 513];

cfg.gt = struct('name', 'gt', ...
                'dir', fullfile(cfg.root, 'hand selected'), ...
                'times_file', 'times_mLFUS08 raw_328.mat', ...
                'grapes_file', 'grapes_blanks.mat', ...
                'class', 1);

cfg.arms = struct('name',        {'python_norefract_sd1', 'python_norefract_sd2'}, ...
                  'dir',         {fullfile(cfg.root, 'python_rescue_no_refract_sd1'), ...
                                  fullfile(cfg.root, 'python_rescue_no_refract_sd2')}, ...
                  'times_file',  {'times_mLFUS08 raw_328.mat', 'times_mLFUS08 raw_328.mat'}, ...
                  'grapes_file', {'grapes_blanks_rescue.mat', 'grapes_blanks_rescue.mat'}, ...
                  'class',       {[], []});

cfg.out_dir = fullfile(cfg.root, 'validation_gt_vs_python_norefract_sd1_sd2');

cfg.tol        = 0.5;
cfg.min_recall = 0.5;
cfg.dyn_buffer = 30;
cfg.x_jitter   = 0.12;
end
