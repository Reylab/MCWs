function cfg = validation_cfg_pt11_emu17_ch328_wave_clus_vs_python()
% VALIDATION_CFG_PT11_EMU17_CH328_WAVE_CLUS_VS_PYTHON  pt 11 / EMU 17 / ch328: wave_clus (reference) vs Python second pass, sd1 rescue (arm)
%   reference ("GT" side): wave_clus class 1 (stock wave_clus-master, raw NC5
%     detection 2026-10-08 23:53, GUI clustering; wave_clus_20261008/, built by build_wave_clus_arm)
%   arm: Python second pass of 2026-10-08 (python_recluster_20261008_second_pass) -- as the arm so its quarantine / rescue / forcing / Mahalanobis show
cfg = validation_cfg_pt11_emu17_ch328();
cfg.gt = struct('name', 'wave_clus', 'dir', fullfile(cfg.root, 'wave_clus_20261008'), ...
                'times_file', 'times_mLFUS08 raw_328.mat', 'grapes_file', 'grapes_blanks.mat', 'class', 1);
cfg.arms = struct('name', 'recluster_second_pass', 'dir', fullfile(cfg.root, 'python_recluster_20261008_second_pass'), ...
                  'times_file', 'times_mLFUS08 raw_328.mat', 'grapes_file', 'grapes_blanks_rescue.mat', 'class', []);
cfg.out_dir = fullfile(cfg.root, 'validation_wave_clus_vs_python_sd1_20261008');
cfg.venn_detections = true;   % every wave_clus spike vs every Python spike
cfg.channel_subsets = true;   % det_* subsets (detection differences only)
end
