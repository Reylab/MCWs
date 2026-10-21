Pp = '/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline';
addpath(fullfile(Pp, 'real_response_analysis')); addpath(fullfile(Pp, 'responses'));
addpath(genpath(fullfile(Pp, 'codes_emu'))); addpath(genpath(fullfile(Pp, 'useful_functions'))); addpath(genpath(fullfile(Pp, 'batch_files')));
R = '/mnt/data0/MCWs Sessions/matlab validation/single channel/pt 11 emu 17 ch 328';
nm = 'times_mLFUS08 raw_328.mat';
new_src = {'/mnt/data0/python_p11EMU17CH/rescue_no_refract_viol_sd1/times_2026-10-05_21-40-34', ...
           '/mnt/data0/python_p11EMU17CH/rescue_no_refract_viol_sd2/times_2026-10-05_21-41-06'};
old_arm = {'python_rescue_test', 'python_rescue_test_sd2'};
new_arm = {'python_rescue_no_refract_sd1', 'python_rescue_no_refract_sd2'};
g = load(fullfile(R, old_arm{1}, 'grapes_blanks.mat'), 'exp_type'); exp_type = g.exp_type;
fprintf('exp_type from old arm: %s\n', exp_type);
ok = true;
for k = 1:2
    tn = load(fullfile(new_src{k}, nm), 'cluster_class', 'quarantine', 'par', 'index');
    to = load(fullfile(R, old_arm{k}, nm), 'cluster_class', 'quarantine', 'index');
    pre_n = tn.cluster_class(:,1) .* ~logical(tn.quarantine.rescued(:));
    pre_o = to.cluster_class(:,1) .* ~logical(to.quarantine.rescued(:));
    same = isequal(pre_n, pre_o) && isequal(tn.cluster_class(:,2), to.cluster_class(:,2));
    fprintf('CHECK %s vs %s: same pre-rescue labels+times=%d | sdnum=%g rescue_refractory=%d | rescued refractory=%d\n', ...
        new_arm{k}, old_arm{k}, same, tn.par.template_sdnum, tn.par.rescue_refractory, ...
        nnz(logical(tn.quarantine.rescued(:)) & logical(tn.quarantine.refractory_period(:))));
    ok = ok && same && tn.par.template_sdnum == k;
end
assert(ok, 'pre-rescue check failed -- not converting');
for k = 1:2
    assert(~exist(fullfile(R, new_arm{k}), 'dir'), '%s already exists', new_arm{k});
    arm = build_python_arm_from_times('root', R, 'channel', 328, 'exp_type', exp_type, ...
        'arm_name', new_arm{k}, 'python_times_file', fullfile(new_src{k}, nm));
    fprintf('CONVERTED %s -> %s\n', new_arm{k}, arm.dir);
end
