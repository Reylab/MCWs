S = '/tmp/claude-1017/-home-ajb-Documents-GitHub-MCWs-MATLAB-pipeline/1d8fd482-04b6-49a4-9526-6897f90343d6/scratchpad/resc_test';
addpath(genpath('/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline/batch_files'));
addpath(genpath('/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline/codes_emu'));
rng(0); t = -60*exp(-0.5*(((1:64)-20)/2).^2);
ng = 200; nc = 12;
spikes_all = [t + 3*randn(ng,64); t + randn(nc,64)];
index_all = sort(rand(1, ng+nc))*1e6;
good = [true(1,ng) false(1,nc)];
mask_non_quarantine = true(1,ng+nc); mask_non_refract = true(1,ng+nc); mask_nonart = true(1,ng+nc);
mask_non_quarantine(ng+(1:4)) = false;          % within-channel only
mask_non_refract(ng+(5:8)) = false;             % refractory only
mask_non_quarantine(ng+(9:12)) = false; mask_non_refract(ng+(9:12)) = false;  % both
save(fullfile(S,'spikes_x','ch_x_spikes.mat'), 'spikes_all','index_all','mask_non_quarantine','mask_non_refract','mask_nonart');
for mode = 1:2
    spikes = spikes_all(good,:); inspk = spikes(:,1:10); cluster_class = [ones(ng,1) index_all(good)'];
    save(fullfile(S,'times_x','times_ch_x.mat'), 'spikes','inspk','cluster_class');
    cd(S);
    if mode == 1
        rescue_spikes('all','folder_name','times_x','spikes_folder',fullfile(S,'spikes_x'),'sdnum',3,'min_spikes',1);
    else
        rescue_spikes('all','folder_name','times_x','spikes_folder',fullfile(S,'spikes_x'),'sdnum',3,'min_spikes',1,'rescue_refractory',true);
    end
    R = load(fullfile(S,'times_x','times_ch_x.mat'),'rescue_mask');
    fprintf('MODE %d rescued candidates (1-4 wc, 5-8 refr, 9-12 both): %s\n', mode, mat2str(find(R.rescue_mask) - ng));
end
