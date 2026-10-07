%% MEMBERSHIP ANALYSIS -- editable test-run script (not a function)
% Edit the CONFIG block below per session/channel/trial-set, then just run
% the file. Everything is a plain variable in the base workspace, so you can
% also run it cell-by-cell (Ctrl+Enter) and inspect/tweak `masked_dir`,
% `rescue_trials`, `acm_out`, etc. by hand between sections.
%
% Does NOT call validate_recluster_vs_handselect.m -- that tool also builds
% the handselect arm from a raster reconstruction, runs
% characterize_grapes_responses, sorting metrics, class matching and a
% report, none of which this needs, and it's slow. Instead build_masked_arm.m
% does ONLY the masking+clustering steps, mirroring that tool's pattern
% (cd(root), copy newest spikes*, within_channel, refract_viol, Do_features,
% Do_clustering, snapshot) directly. The ground truth is used as-is -- its
% grapes_blanks.mat needs no processing at all, just a path.
%
% Pipeline: build_masked_arm -> build_rescue_sd_trials -> mask_audit
%           -> analyze_cluster_membership
%
% ALL output lands under ONE folder, ORGANIZED:
%   MEMBERSHIP_ch<ch>_<ts>/
%     masked_arm/           <- build_masked_arm.m
%     mask_audit/          within_channel + refract_viol per-check breakdown
%     rescue_sd_sweep/
%       rescue_sd1/  rescue_sd2/  rescue_sd3/   (spikes+times+grapes_blanks_rescue each)
%     membership/  response/  mu_window/  rasters/   <- analyze_cluster_membership

clear; clc;
addpath(genpath('/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline'));

%% ======================= CONFIG -- edit per run ==========================
root = '/mnt/data0/MCWs Sessions/matlab validation/45 chan (incomplete bundles)/compare handselect/pt 12 emu 18 ch 300';
ch   = 300;

% hand-select ground truth: just a path, no processing. Must be blanks-flavour.
ref_grapes = fullfile(root, 'hand selected', 'grapes_blanks.mat');
ref_name   = 'handselect';

sdnums      = [1 2 3];    % second-pass rescue sdnums to sweep (first pass fixed at 1)

% ONE ground-truth class only -- this is the whole design, not a default to
% change lightly. A scalar. Everything (membership, response metrics, change
% summary, rasters) runs for THIS class and this class alone.
gt_class    = 1;

% ORDERE3 stimulus list for the raster stage -- YOUR order, verbatim, 20-40
% at a time (see analyze_cluster_membership's 'stim_list'). Required for the
% rasters to mean anything; omitting it falls back to first-seen order with
% a warning.
stim_list = [94 24 12 22 16 1 8 85 13 80];

within_channel_detections = 'all';    % forwarded to build_masked_arm
refract_keep_strategy     = 'none';   % "
exp_type    = 'RSVP_online';

cd(root);   % within_channel/refract_viol/Do_clustering all auto-detect off pwd

% capture the newest spikes* BEFORE build_masked_arm makes its own -- hand-
% select's cluster_class is curated from THIS spike set (the times_*.mat in
% 'hand selected' has no matching *_spikes.mat of its own), needed for its
% spike-quality metrics (compute_metrics_batch).
d0 = dir(fullfile(root, 'spikes*')); d0 = d0([d0.isdir]);
assert(~isempty(d0), 'no spikes* folder under %s', root);
[~, i0] = max([d0.datenum]);
ref_spikes_dir = fullfile(root, d0(i0).name);

%% ======================= organized output folder =========================
ts = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
out_root = fullfile(root, sprintf('MEMBERSHIP_ch%d_%s', ch, ts));
mkdir(out_root);
dir_masked = fullfile(out_root, 'masked_arm');
dir_mask   = fullfile(out_root, 'mask_audit');
dir_sweep  = fullfile(out_root, 'rescue_sd_sweep');
fprintf('[run] output root: %s\n', out_root);

%% ======================= 1. build the masked arm, fresh ===================
masked_dir = build_masked_arm(root, ch, exp_type, ...
    'within_channel_detections', within_channel_detections, ...
    'refract_keep_strategy', refract_keep_strategy, ...
    'out_dir', dir_masked);

NSx = getfield(load(fullfile(root, 'NSx.mat')), 'NSx'); 
nm  = NSx(arrayfun(@(x) x.chan_ID==ch, NSx)).output_name;

%% ======================= 2. rescue sd sweep, off the just-built masked arm
rescue_trials = build_rescue_sd_trials(masked_dir, root, ch, exp_type, ...
    'sdnums', sdnums, 'out_dir', dir_sweep);

%% ======================= 3. mask audit ====================================
masked_spikes_file = fullfile(masked_dir, [nm '_spikes.mat']);
masked_times_file  = fullfile(masked_dir, ['times_' nm '.mat']);
mask_out = mask_audit(masked_spikes_file, 'times_file', masked_times_file, ...
    'rescue_trials', rescue_trials, 'out_dir', dir_mask);

%% ======================= 4. assemble trials + compare =====================
% Post-rescue only: handselect vs rescue_sd1/sd2/sd3. The pre-rescue
% 'masked' state is used above (build_rescue_sd_trials sweeps off it,
% mask_audit inspects it) but is NOT its own comparison arm here.
trials = struct('name', {}, 'grapes_file', {}, 'times_file', {}, 'spikes_file', {});
ref_times_file  = fullfile(root, 'hand selected', ['times_' nm '.mat']);
ref_spikes_file = fullfile(ref_spikes_dir, [nm '_spikes.mat']);
trials(end+1) = struct('name', ref_name, 'grapes_file', ref_grapes, ...
                       'times_file', ref_times_file, 'spikes_file', ref_spikes_file);
for i = 1:numel(rescue_trials)
    trials(end+1) = struct('name', rescue_trials(i).name, 'grapes_file', rescue_trials(i).grapes_file, ...
                           'times_file', rescue_trials(i).times_file, 'spikes_file', rescue_trials(i).spikes_file);
end

acm_out = analyze_cluster_membership(trials, ref_name, ch, 'root', root, ...
    'gt_classes', gt_class, 'stim_list', stim_list, 'out_dir', out_root, ...
    'masked_spikes_file', masked_spikes_file);

fprintf('\n[run] DONE. Everything under:\n  %s\n', out_root);
disp(acm_out.membership);
