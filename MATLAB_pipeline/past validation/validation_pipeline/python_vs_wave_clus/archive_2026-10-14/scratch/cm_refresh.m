addpath('/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline/validation_pipeline');
SC = '/mnt/data0/MCWs Sessions/matlab validation/single channel/';
V = {'pt 6 emu 13 ch 333/wave_clus4/validation_wave_clus4_vs_python_sd1', 'pt 4 emu 7 ch 259/wave_clus2/validation_wave_clus2_vs_python_sd1', ...
     'pt 12 emu 18 ch 299/wave_clus7/validation_wave_clus7_vs_python_sd1', 'pt 12 emu 18 ch 300/wave_clus4/validation_wave_clus4_vs_python_sd1_class1', ...
     'pt 12 emu 18 ch 300/wave_clus4/validation_wave_clus4_vs_python_sd1_class2'};
for v = V
    s = load(fullfile(SC, v{1}, 'validation_state.mat'), 'cfg'); cfg = s.cfg; cfg.stages = {'class_match'};
    run_validation(cfg);
    d = dir(fullfile(SC, v{1}, 'class_match_*.png')); fprintf('[cm] %s: python class %d -> %s\n', v{1}, cfg.arms.class, strjoin({d.name}, ', '));
end
