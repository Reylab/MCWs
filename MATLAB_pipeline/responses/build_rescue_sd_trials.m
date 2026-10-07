function trials = build_rescue_sd_trials(masked_dir, root, ch, exp_type, varargin)
% BUILD_RESCUE_SD_TRIALS  Build sd1/sd2/sd3 rescue "trials" that share ONE
% masked clustering (no re-SPC) and differ only in the second-pass
% force_membership_wc template radius. First pass is fixed at sd1 (rescue's
% own default); second pass is swept.
%
% Each sdnum gets its own COPY of the masked arm's spikes+times files, is
% rescued independently (not chained), then has its OWN grapes_blanks +
% grapes_blanks_rescue built -- so sd1/sd2/sd3 are three independent,
% directly-comparable trials, exactly the shape analyze_cluster_membership
% expects.
%
% USAGE
%   trials = build_rescue_sd_trials(masked_dir, root, ch, exp_type)
%   trials = build_rescue_sd_trials(masked_dir, root, ch, exp_type, 'sdnums', [1 2 3])
%
% INPUTS
%   masked_dir   folder with times_<nm>.mat + <nm>_spikes.mat for one
%                already-clustered masked arm (e.g. .../masked_recluster)
%   root         session root (NSx.mat, stimulus.mat, finalevents.mat, ...)
%   ch           channel number
%   exp_type     passed straight to do_structure_mu_BCM_online3
%
%   Name/value:
%     'sdnums'           [1 2 3]   second-pass template_sdnum values to test
%     'first_pass_sdnum' 1         held fixed across all sdnums (rescue's own
%                                   default -- passed through explicitly so
%                                   it's visible/auditable here)
%     'circshiftblanks'  false
%     'out_dir'          ''        default: <root>/rescue_sd_sweep_<ts>/
%
% OUTPUT
%   trials   struct array, one per sdnum, each with:
%              .name         'rescue_sd1' / 'rescue_sd2' / 'rescue_sd3'
%              .sdnum
%              .dir          the trial's folder
%              .grapes_file  .../grapes_blanks_rescue.mat (post-rescue classN)
%              .times_file   .../times_<nm>.mat (post-rescue cluster_class)
%              .spikes_file  .../<nm>_spikes.mat
%
% Feed `trials` (plus a 'masked' trial pointing at masked_dir itself, and a
% reference trial) into analyze_cluster_membership.

    p = inputParser;
    addParameter(p, 'sdnums', [1 2 3]);
    addParameter(p, 'first_pass_sdnum', 1);
    addParameter(p, 'circshiftblanks', false);
    addParameter(p, 'out_dir', '');
    parse(p, varargin{:});
    o = p.Results;

    addpath(genpath('/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline'));

    NSx = getfield(load(fullfile(root, 'NSx.mat')), 'NSx');
    pos = find(arrayfun(@(x) x.chan_ID == ch, NSx), 1);
    assert(~isempty(pos), 'channel %d not found in NSx.mat', ch);
    nm = NSx(pos).output_name;

    sf = fullfile(masked_dir, [nm '_spikes.mat']);
    tf = fullfile(masked_dir, ['times_' nm '.mat']);
    assert(exist(sf,'file')==2, 'missing %s', sf);
    assert(exist(tf,'file')==2, 'missing %s', tf);

    if isempty(o.out_dir)
        ts = char(datetime('now','Format','yyyyMMdd_HHmmss'));
        o.out_dir = fullfile(root, ['rescue_sd_sweep_' ts]);
    end
    if ~exist(o.out_dir, 'dir'), mkdir(o.out_dir); end

    trials = struct('name', {}, 'sdnum', {}, 'dir', {}, 'grapes_file', {}, ...
                    'times_file', {}, 'spikes_file', {});

    cd(root);
    for si = 1:numel(o.sdnums)
        k = o.sdnums(si);
        name = sprintf('rescue_sd%g', k);
        D = fullfile(o.out_dir, name);
        if ~exist(D, 'dir'), mkdir(D); end
        copyfile(sf, fullfile(D, [nm '_spikes.mat']));
        copyfile(tf, fullfile(D, ['times_' nm '.mat']));
        tmpl = fullfile(masked_dir, 'templates_wc_offline.mat');
        if exist(tmpl, 'file'), copyfile(tmpl, fullfile(D, 'templates_wc_offline.mat')); end

        fprintf('[rescue_sd_sweep] %s: sdnum=%g (first_pass_sdnum=%g, fixed)\n', ...
                name, k, o.first_pass_sdnum);
        rel_folder = erase(D, [root filesep]);   % 'folder_name' is relative to pwd (=root)
        rescue_spikes(ch, 'restore', false, 'parallel', false, ...
                      'folder_name', rel_folder, 'spikes_folder', D, ...
                      'sdnum', k, 'first_pass_sdnum', o.first_pass_sdnum);

        do_structure_mu_BCM_online3(ch, exp_type, true, o.circshiftblanks, false, 'spike_dir', D);
        do_structure_sorted_BCM_online3(ch, true, o.circshiftblanks, false, ...
                                        'times_dir', D, 'spike_dir', D);
        do_structure_rescue_BCM_online3(ch, true, o.circshiftblanks, false, ...
                                        'times_dir', D, 'spike_dir', D);

        gf = fullfile(D, 'grapes_blanks_rescue.mat');
        assert(exist(gf,'file')==2, '%s: expected %s, not written', name, gf);

        trials(end+1) = struct('name', name, 'sdnum', k, 'dir', D, ...
            'grapes_file', gf, 'times_file', fullfile(D, ['times_' nm '.mat']), ...
            'spikes_file', fullfile(D, [nm '_spikes.mat']));
    end
    fprintf('[rescue_sd_sweep] %d trials in %s\n', numel(trials), o.out_dir);
end
