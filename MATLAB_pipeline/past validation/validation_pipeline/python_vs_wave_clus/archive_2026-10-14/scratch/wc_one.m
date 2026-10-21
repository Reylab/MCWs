addpath('/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline/validation_pipeline');
root = '/mnt/data0/MCWs Sessions/matlab validation/single channel/pt 12 emu 18 ch 299';
wc = fullfile(root, 'wave_clus1'); f = fullfile(wc, 'times_mRFUS02 raw_299.mat');
before = load(f); fb = fieldnames(before);
build_wave_clus_arm(root, 299, wc);
after = load(f);
same = all(cellfun(@(k) isequaln(before.(k), after.(k)), fb));
fprintf('original fields unchanged: %d | added: %s\n', same, strjoin(setdiff(fieldnames(after), fb)', ', '));
disp(ls(wc))
