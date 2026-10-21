function cfg = validation_cfg_pt4_emu7_ch259()
% VALIDATION_CFG_PT4_EMU7_CH259  run_validation config: pt 4 / EMU 7 / ch259, hand-selected GT
% (class 1) vs the Python recluster run of 2026-10-07
% (/mnt/data0/recluster_runs/p04_EMU-007_ch259_selectivity), converted by convert_recluster_runs:
%   python_recluster_20261007_rescued - clustering + artifact rescue (the full set)
%
% stim_list as given, minus a repeated 228 (it was listed 1st and 11th).

cfg.root = '/mnt/data0/MCWs Sessions/matlab validation/single channel/pt 4 emu 7 ch 259';
cfg.ch = 259;
cfg.stim_list = [228 294 163 45 155 60 84 122 131 79 106 267 318 178 107 72 283 196 43];   % your list has 228 twice (1st and 11th); listed once

cfg.gt = struct('name', 'gt', ...
                'dir', fullfile(cfg.root, 'hand selected'), ...
                'times_file', 'times_mLPBTG03 raw_259.mat', ...
                'grapes_file', 'grapes_blanks.mat', ...
                'class', 1);

cfg.arms = struct('name',        'recluster_rescued', ...
                  'dir',         fullfile(cfg.root, 'python_recluster_20261007_rescued'), ...
                  'times_file',  'times_mLPBTG03 raw_259.mat', ...
                  'grapes_file', 'grapes_blanks_rescue.mat', ...
                  'class',       []);

cfg.out_dir = fullfile(cfg.root, 'validation_gt_vs_python_recluster_20261007');

cfg.tol        = 0.5;   % ms, spike-time match tolerance
cfg.min_recall = 0.5;   % match_cc threshold for calling an arm class "matched"
cfg.dyn_buffer = 30;    % ms on each side of onset..onset+dura (green window lines)
cfg.x_jitter   = 0.12;  % half-width of x jitter on the GT-vs-ARM paired-line panels
end
