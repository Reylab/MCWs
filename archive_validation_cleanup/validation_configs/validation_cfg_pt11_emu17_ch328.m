function cfg = validation_cfg_pt11_emu17_ch328()
% VALIDATION_CFG_PT11_EMU17_CH328  run_validation config: pt 11 / EMU 17 / ch328, hand-selected GT
% (class 1) vs the Python recluster run of 2026-10-07
% (/mnt/data0/recluster_runs/p11_EMU-017_ch328), converted by convert_recluster_runs:
%   python_recluster_20261008_second_pass - the finished second pass of 2026-10-08
%   16:48 (recluster_runs/p11_EMU-017_ch328/second_pass/times_2026-10-08_16-48-45),
%   the full set; it saves per-spike mahal_reassigned

cfg.root = '/mnt/data0/MCWs Sessions/matlab validation/single channel/pt 11 emu 17 ch 328';
cfg.ch = 328;
cfg.stim_list = [76 386 858 69 391 875 139 1015 513];

cfg.gt = struct('name', 'gt', ...
                'dir', fullfile(cfg.root, 'hand selected'), ...
                'times_file', 'times_mLFUS08 raw_328.mat', ...
                'grapes_file', 'grapes_blanks.mat', ...
                'class', 1);

cfg.arms = struct('name',        'recluster_second_pass', ...
                  'dir',         fullfile(cfg.root, 'python_recluster_20261008_second_pass'), ...
                  'times_file',  'times_mLFUS08 raw_328.mat', ...
                  'grapes_file', 'grapes_blanks_rescue.mat', ...
                  'class',       []);

cfg.out_dir = fullfile(cfg.root, 'validation_gt_vs_python_recluster_20261008_second_pass');
cfg.venn_detections = true;   % also the all-detections venn (GT vs arm detectors)
cfg.channel_subsets = true;   % also save full-venn and detections-venn subsets (whole channel)

cfg.tol        = 0.5;   % ms, spike-time match tolerance
cfg.min_recall = 0.5;   % match_cc threshold for calling an arm class "matched"
cfg.dyn_buffer = 30;    % ms on each side of onset..onset+dura (green window lines)
cfg.x_jitter   = 0.12;  % half-width of x jitter on the GT-vs-ARM paired-line panels
end
