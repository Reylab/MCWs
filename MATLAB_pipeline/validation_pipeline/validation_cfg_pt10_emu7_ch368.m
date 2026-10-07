function cfg = validation_cfg_pt10_emu7_ch368()
% VALIDATION_CFG_PT10_EMU7_CH368  Config for run_validation: pt 10 / EMU 7 /
% ch368, hand-selected GT (class 3) vs the Python rescue tests at sd1 and sd2.
%
% Copy this file to set up another session/channel -- run_validation itself
% has nothing dataset-specific in it.

cfg.root = '/mnt/data0/MCWs Sessions/matlab validation/single channel/pt 10 emu 7 ch 368';
cfg.ch = 368;
cfg.stim_list = [410 330 532 776 492 496 72 124 951 130 166 273 585 35 533 16 542 3 490 180];

% Ground truth. class is required (it's what the arms get matched against).
cfg.gt = struct('name', 'gt', ...
                'dir', fullfile(cfg.root, 'hand selected'), ...
                'times_file', 'times_mRFUS07 raw_368.mat', ...
                'grapes_file', 'grapes_blanks.mat', ...
                'class', 3);

% Arms to compare against GT. class = [] -> matched to the GT class by spike
% time (match_arms_to_ground_truth); set a number to force it.
cfg.arms = struct('name',        {'python_sd1', 'python_sd2'}, ...
                  'dir',         {fullfile(cfg.root, 'python_rescue_test_sd1'), ...
                                  fullfile(cfg.root, 'python_rescue_test_sd2')}, ...
                  'times_file',  {'times_mRFUS07 raw_368.mat', 'times_mRFUS07 raw_368.mat'}, ...
                  'grapes_file', {'grapes_blanks_rescue.mat', 'grapes_blanks_rescue.mat'}, ...
                  'class',       {[], []});

cfg.out_dir = fullfile(cfg.root, 'validation_gt_vs_python_sd1_sd2');

cfg.tol        = 0.5;   % ms, spike-time match tolerance
cfg.min_recall = 0.5;   % match_cc threshold for calling an arm class "matched"
cfg.dyn_buffer = 30;    % ms on each side of onset..onset+dura (green window lines)
cfg.x_jitter   = 0.12;  % half-width of x jitter on the GT-vs-ARM paired-line panels
end
