function cfg = validation_cfg_pt10_emu7_ch368_wave_clus_vs_python()
% VALIDATION_CFG_PT10_EMU7_CH368_WAVE_CLUS_VS_PYTHON  pt 10 / EMU 7 / ch368: wave_clus (reference) vs Python recluster sd1 (arm)
%   reference ("GT" side): wave_clus class 1 (stock wave_clus-master, raw NC5
%     detection 2026-10-08 23:45, GUI clustering; wave_clus_20261008/, built by build_wave_clus_arm)
%   arm: Python recluster sd1 (python_recluster_20261006_sd1) -- as the arm so its quarantine / rescue / forcing show
cfg = validation_cfg_pt10_emu7_ch368();
cfg.gt = struct('name', 'wave_clus', 'dir', fullfile(cfg.root, 'wave_clus_20261008'), ...
                'times_file', 'times_mRFUS07 raw_368.mat', 'grapes_file', 'grapes_blanks.mat', 'class', 1);
cfg.arms = struct('name', 'python_recluster_sd1', 'dir', fullfile(cfg.root, 'python_recluster_20261006_sd1'), ...
                  'times_file', 'times_mRFUS07 raw_368.mat', 'grapes_file', 'grapes_blanks_rescue.mat', 'class', []);
cfg.out_dir = fullfile(cfg.root, 'validation_wave_clus_vs_python_sd1_20261008');
cfg.venn_detections = true;   % every wave_clus spike vs every Python spike
cfg.channel_subsets = true;   % det_* subsets (detection differences only)
end
