function cfg = validation_cfg_pt9_emu3_ch335()
% VALIDATION_CFG_PT9_EMU3_CH335  run_validation config: pt 9 / EMU 3 / ch335, hand-selected GT
% (class 2) vs the Python recluster run of 2026-10-07
% (/mnt/data0/recluster_runs/p09_EMU-003_ch335_family-resp), converted by convert_recluster_runs:
%   python_recluster_20261007_rescued - clustering + artifact rescue (the full set)

cfg.root = '/mnt/data0/MCWs Sessions/matlab validation/single channel/zneed rerun grapes blanks changednew trial pt 9 emu 3 ch 335';
cfg.ch = 335;
cfg.stim_list = [288 156 272 278 55 147 71 12 64 282 146 25 7 228 83 615 275 482 63 46];

cfg.gt = struct('name', 'gt', ...
                'dir', fullfile(cfg.root, 'hand selected'), ...
                'times_file', 'times_mLFUS06 raw_335.mat', ...
                'grapes_file', 'grapes_blanks.mat', ...
                'class', 2);

cfg.arms = struct('name',        'recluster_rescued', ...
                  'dir',         fullfile(cfg.root, 'python_recluster_20261007_rescued'), ...
                  'times_file',  'times_mLFUS06 raw_335.mat', ...
                  'grapes_file', 'grapes_blanks_rescue.mat', ...
                  'class',       []);

cfg.out_dir = fullfile(cfg.root, 'validation_gt_vs_python_recluster_20261007');

cfg.tol        = 0.5;   % ms, spike-time match tolerance
cfg.min_recall = 0.5;   % match_cc threshold for calling an arm class "matched"
cfg.dyn_buffer = 30;    % ms on each side of onset..onset+dura (green window lines)
cfg.x_jitter   = 0.12;  % half-width of x jitter on the GT-vs-ARM paired-line panels
end
