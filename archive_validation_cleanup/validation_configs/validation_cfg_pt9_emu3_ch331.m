function cfg = validation_cfg_pt9_emu3_ch331()
% VALIDATION_CFG_PT9_EMU3_CH331  run_validation config: pt 9 / EMU 3 / ch331, hand-selected GT
% (class 4) vs the Python recluster run of 2026-10-07
% (/mnt/data0/recluster_runs/p09_EMU-003_ch331_vacation-neuron), converted by convert_recluster_runs:
%   python_recluster_20261007_rescued - clustering + artifact rescue (the full set)

cfg.root = '/mnt/data0/MCWs Sessions/matlab validation/single channel/pt 9 emu 3 ch 331 check';
cfg.ch = 331;
cfg.stim_list = [96 176 43 324 414 744 482 204 785 250 236 12 78 95 503 177 125 196 208 230];

% GT: the 4-class sort the hand-selected grapes were built from, rebuilt from
% its raster (class 4 = 2710 of the original 3538; the between-trial spikes are
% missing). The 14-class times file in 'hand selected' is a different sort.
cfg.gt = struct('name', 'gt', ...
                'dir', fullfile(cfg.root, 'hand selected rebuilt 4class'), ...
                'times_file', 'times_mLFUS02 raw_331.mat', ...
                'grapes_file', 'grapes_blanks.mat', ...
                'class', 4);

cfg.arms = struct('name',        'recluster_rescued', ...
                  'dir',         fullfile(cfg.root, 'python_recluster_20261007_rescued'), ...
                  'times_file',  'times_mLFUS02 raw_331.mat', ...
                  'grapes_file', 'grapes_blanks_rescue.mat', ...
                  'class',       []);

cfg.out_dir = fullfile(cfg.root, 'validation_gt_vs_python_recluster_20261007');

cfg.tol        = 0.5;   % ms, spike-time match tolerance
cfg.min_recall = 0.5;   % match_cc threshold for calling an arm class "matched"
cfg.dyn_buffer = 30;    % ms on each side of onset..onset+dura (green window lines)
cfg.x_jitter   = 0.12;  % half-width of x jitter on the GT-vs-ARM paired-line panels
end
