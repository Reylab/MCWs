addpath('/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline/validation_pipeline');
root = '/mnt/data0/MCWs Sessions/matlab validation/single channel/pt 10 emu 7 ch 368';
for w = 1:10
    try
        build_wave_clus_arm(root, 368, fullfile(root, sprintf('wave_clus%d', w)));
    catch ME
        fprintf('[wave_clus arm] FAILED wave_clus%d: %s\n', w, ME.message);
    end
end
