% PPT_PT4_EMU7_CH259  for ppt report of pt 4 emu 7 ch 259 (GT c1): wave_clus notch vs python vs GT
% (see build_ppt_report for the layout). Channel-specific extra analysis goes below, using M.
addpath('/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline/validation_pipeline');
o.session  = 'pt4_ch259';
o.runs     = struct('label', 'notch', 'folder', 'wave_clus notch');
o.no_notch = 'wave_clus no notch';
M = build_ppt_report(o);
