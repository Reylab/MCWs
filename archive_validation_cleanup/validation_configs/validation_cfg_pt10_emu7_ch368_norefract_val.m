function cfg = validation_cfg_pt10_emu7_ch368_norefract_val()
% VALIDATION_CFG_PT10_EMU7_CH368_NOREFRACT_VAL  Same as
% validation_cfg_pt10_emu7_ch368_norefract (left as-is for its own testing
% output folder), but writing to the standard validation output folder.

cfg = validation_cfg_pt10_emu7_ch368_norefract();
cfg.out_dir = fullfile(cfg.root, 'validation_gt_vs_python_norefract_sd1_sd2');
end
