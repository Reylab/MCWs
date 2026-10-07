function [Gs, yl, times_arms, arm_classes] = prepare_arm_groups(cfg, arm_classes)
% PREPARE_ARM_GROUPS  Load every arm, resolve its matched class (unless
% given), build its waveform groups (arm_wave_groups) and the single fixed
% y range shared by every waveform panel of every arm (wave_ylim).
    here = fileparts(mfilename('fullpath'));
    addpath(fullfile(fileparts(here), 'real_response_analysis'));
    addpath(fullfile(fileparts(here), 'responses'));

    arms = cfg.arms;
    times_arms = cell(1, numel(arms));
    for k = 1:numel(arms)
        times_arms{k} = load(fullfile(arms(k).dir, arms(k).times_file));
    end
    times_gt = load(fullfile(cfg.gt.dir, cfg.gt.times_file), 'cluster_class');
    if nargin < 2 || isempty(arm_classes)
        arm_classes = resolve_arm_classes(cfg, times_gt, times_arms);
    end
    gt_t = times_gt.cluster_class(times_gt.cluster_class(:,1) == cfg.gt.class, 2);

    Gs = cell(1, numel(arms));
    for k = 1:numel(arms)
        Gs{k} = arm_wave_groups(times_arms{k}, arm_classes(k), gt_t, cfg.tol);
    end
    yl = wave_ylim(Gs);
end
