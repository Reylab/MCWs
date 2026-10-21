function cfg = validation_cfg_pt12_emu18_ch305()
% VALIDATION_CFG_PT12_EMU18_CH305  run_validation config: pt 12 / EMU 18 / ch305, hand-selected GT
% (class 1) vs the Python recluster run of 2026-10-07
% (/mnt/data0/recluster_runs/p12_EMU-018_ch305_numb), converted by convert_recluster_runs:
%   python_recluster_20261007_rescued - clustering + artifact rescue (the full set)

cfg.root = '/mnt/data0/MCWs Sessions/matlab validation/single channel/pt 12 emu 18 ch 305';
cfg.ch = 305;
cfg.stim_list = [215 198 209 211 210];

cfg.gt = struct('name', 'gt', ...
                'dir', fullfile(cfg.root, 'hand selected'), ...
                'times_file', 'times_mRFUS08 raw_305.mat', ...
                'grapes_file', 'grapes_blanks.mat', ...
                'class', 1);

cfg.arms = struct('name',        'recluster_rescued', ...
                  'dir',         fullfile(cfg.root, 'python_recluster_20261007_rescued'), ...
                  'times_file',  'times_mRFUS08 raw_305.mat', ...
                  'grapes_file', 'grapes_blanks_rescue.mat', ...
                  'class',       []);

cfg.out_dir = fullfile(cfg.root, 'validation_gt_vs_python_recluster_20261007');

cfg.tol        = 0.5;   % ms, spike-time match tolerance
cfg.min_recall = 0.5;   % match_cc threshold for calling an arm class "matched"
cfg.dyn_buffer = 30;    % ms on each side of onset..onset+dura (green window lines)
cfg.x_jitter   = 0.12;  % half-width of x jitter on the GT-vs-ARM paired-line panels
end
