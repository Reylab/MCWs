addpath('/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline/validation_pipeline');
SC = '/mnt/data0/MCWs Sessions/matlab validation/single channel/';
J = {'pt 6 emu 13 ch 333', 333, 1:5; 'pt 4 emu 7 ch 259', 259, 1:5; 'pt 12 emu 18 ch 300', 300, [1 3 4]; ...
     'pt 12 emu 18 ch 299', 299, 2:8};
for i = 1:size(J, 1)
    for w = J{i, 3}
        root = [SC J{i, 1}]; wc = fullfile(root, sprintf('wave_clus%d', w));
        try
            build_wave_clus_arm(root, J{i, 2}, wc);
        catch ME
            fprintf('[wave_clus arm] FAILED %s wave_clus%d: %s\n', J{i, 1}, w, ME.message);
        end
    end
end
