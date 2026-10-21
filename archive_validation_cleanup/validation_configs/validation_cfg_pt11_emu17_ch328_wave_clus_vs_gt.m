function cfg = validation_cfg_pt11_emu17_ch328_wave_clus_vs_gt()
% VALIDATION_CFG_PT11_EMU17_CH328_WAVE_CLUS_VS_GT  pt 11 / EMU 17 / ch328: hand-selected GT (class 1) vs wave_clus
%   arm: wave_clus (stock wave_clus-master, raw NC5 detection 2026-10-08 23:53, GUI clustering),
%        built into wave_clus_20261008/ by build_wave_clus_arm
%   reference: hand-selected GT, class 1
cfg = validation_cfg_pt11_emu17_ch328();
cfg.arms = struct('name', 'wave_clus', 'dir', fullfile(cfg.root, 'wave_clus_20261008'), ...
                  'times_file', 'times_mLFUS08 raw_328.mat', 'grapes_file', 'grapes_blanks.mat', 'class', []);
cfg.out_dir = fullfile(cfg.root, 'validation_gt_vs_wave_clus_20261008');
cfg.venn_detections = true;   % all detections: reference detector vs wave_clus detector
end
