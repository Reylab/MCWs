function cfg = validation_cfg_pt6_emu13_ch333()
% VALIDATION_CFG_PT6_EMU13_CH333  run_validation config: pt 6 / EMU 13 / ch333, hand-selected GT
% (class 1) vs the Python recluster run of 2026-10-07
% (/mnt/data0/recluster_runs/p06_EMU-013_ch333_zuck-ticks-wwe), converted by convert_recluster_runs:
%   python_recluster_20261007_rescued - clustering + artifact rescue (the full set)

cfg.root = '/mnt/data0/MCWs Sessions/matlab validation/single channel/pt 6 emu 13 ch 333';
cfg.ch = 333;
cfg.stim_list = [642 372 589 25 82 96 179 293 436 177 98 647 40 285 613 394 582 626 632 113];

cfg.gt = struct('name', 'gt', ...
                'dir', fullfile(cfg.root, 'hand selected'), ...
                'times_file', 'times_mLFUS04 raw_333.mat', ...
                'grapes_file', 'grapes_blanks.mat', ...
                'class', 1);

cfg.arms = struct('name',        'recluster_rescued', ...
                  'dir',         fullfile(cfg.root, 'python_recluster_20261007_rescued'), ...
                  'times_file',  'times_mLFUS04 raw_333.mat', ...
                  'grapes_file', 'grapes_blanks_rescue.mat', ...
                  'class',       []);

cfg.out_dir = fullfile(cfg.root, 'validation_gt_vs_python_recluster_20261007');

cfg.tol        = 0.5;   % ms, spike-time match tolerance
cfg.min_recall = 0.5;   % match_cc threshold for calling an arm class "matched"
cfg.dyn_buffer = 30;    % ms on each side of onset..onset+dura (green window lines)
cfg.x_jitter   = 0.12;  % half-width of x jitter on the GT-vs-ARM paired-line panels
end
