function patch_tuned_qc_params(chs)
% PATCH_TUNED_QC_PARAMS  Record the tuned trial's within-channel limits in
% par.qc_params of the saved tuned sorts (trial_replay.py saved them with the
% as-run par, width 3-15). MATLAB rebuilds the per-check quarantine flags
% (adapt_python_spikes_for_matlab) from par.qc_params, so with the old value
% the spikes the trial put back (width 15-25) were still counted as "bad
% width". Checked first: with max_width_idx = 25 the rebuilt flags equal the
% tuned quarantine.within_channel on every spike of all six channels.
%
%   patch_tuned_qc_params([259 299 300 328 333 368])
% Only par is rewritten (save -append); every other variable is untouched.
here = fileparts(mfilename('fullpath'));
C = jsondecode(fileread(fullfile(here, 'channels_tuned.json')));
W = C.trial.within_channel;
for u = C.units'
    if ~ismember(u.ch, chs), continue; end
    sorter = dir(fullfile(C.recluster_runs, u.run, C.second_pass_subdir, 'second_pass_*', '*', 'times_*', 'times_*.mat'));
    arm = dir(fullfile(C.single_channel, u.folder, u.python_arm, '*.mat'));
    files = [fullfile({sorter.folder}, {sorter.name}), fullfile({arm.folder}, {arm.name})];
    for f = files
        if ~ismember('par', who('-file', f{1})), continue; end
        S = load(f{1}, 'par'); par = S.par;
        if ~isfield(par, 'qc_params'), continue; end
        old = par.qc_params.max_width_idx;
        par.qc_params.min_width_idx = W.width_lower;
        par.qc_params.max_width_idx = W.width_upper;
        par.qc_params.final_prominence_ratio_pass = W.prom_ratio;
        save(f{1}, 'par', '-append');
        fprintf('ch%d  %s: max_width_idx %g -> %g\n', u.ch, f{1}, old, W.width_upper);
    end
end
end
