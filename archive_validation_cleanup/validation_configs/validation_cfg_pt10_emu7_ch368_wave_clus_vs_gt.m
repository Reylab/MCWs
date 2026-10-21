function cfg = validation_cfg_pt10_emu7_ch368_wave_clus_vs_gt()
% VALIDATION_CFG_PT10_EMU7_CH368_WAVE_CLUS_VS_GT  pt 10 / EMU 7 / ch368: hand-selected GT (class 3) vs wave_clus
%   arm: wave_clus (stock wave_clus-master, raw NC5 detection 2026-10-08 23:45, GUI clustering),
%        built into wave_clus_20261008/ by build_wave_clus_arm
%   reference: hand-selected GT, class 3
cfg = validation_cfg_pt10_emu7_ch368();
cfg.arms = struct('name', 'wave_clus', 'dir', fullfile(cfg.root, 'wave_clus_20261008'), ...
                  'times_file', 'times_mRFUS07 raw_368.mat', 'grapes_file', 'grapes_blanks.mat', 'class', []);
cfg.out_dir = fullfile(cfg.root, 'validation_gt_vs_wave_clus_20261008');
cfg.venn_detections = true;   % all detections: reference detector vs wave_clus detector
end
