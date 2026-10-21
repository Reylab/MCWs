function arm = build_python_arm_from_times(varargin)
% BUILD_PYTHON_ARM_FROM_TIMES  One-off test driver: takes an existing Python
% mcws times_*.mat (already on disk -- NOT re-run here), adapts it to this
% MATLAB pipeline's times_*.mat / *_spikes.mat conventions via
% adapt_python_times_for_matlab.m / adapt_python_spikes_for_matlab.m, then
% runs the EXACT SAME do_structure_mu_BCM_online3 -> do_structure_sorted_
% BCM_online3 -> do_structure_rescue_BCM_online3 recipe build_rescue_sd_
% trials.m uses to build a real rescue arm (rescue_sd1/2/3). The result is a
% normal arm folder (grapes_blanks.mat + grapes_blanks_rescue.mat), usable
% by run_rescue_grape_analysis.m like any other -- none of those three
% do_structure_* functions are modified.
%
% Does NOT run the Python pipeline -- it reads whichever Python times_*.mat
% already exists (the newest one, by folder timestamp, unless overridden).
%
% Name-value args:
%   'root'          session root (NSx.mat/stimulus.mat/finalevents.mat/
%                   experiment_properties_online3.mat) -- default: the real
%                   pt11 EMU17 ch328 session run_rescue_grape_analysis.m uses
%   'python_root'   folder holding the Python mcws times_*/spikes_* output
%                   subfolders -- default /mnt/data0/python_p11EMU17CH
%   'channel'       channel number -- default 328
%   'exp_type'      -- default 'RSVP_online' (read directly from this
%                   session's own real grapes_blanks.mat)
%   'arm_name'      new arm subfolder name created under root -- default
%                   'python_rescue_test'
%   'python_times_file' explicit path to one Python times_*.mat -- default
%                   '' (auto-picks the newest times_* subfolder under
%                   python_root by folder datenum, same "most recent wins"
%                   convention this pipeline's own folder-resolution helpers
%                   use, and looks for times_<output_name>.mat inside it)
%   'circshiftblanks' -- default false
%
% OUTPUT
%   arm - struct with .name, .dir, .grapes_file (grapes_blanks_rescue.mat),
%         .times_file, .spikes_file -- same shape as build_rescue_sd_
%         trials.m's per-trial struct, for drop-in use as a new
%         run_rescue_grape_analysis.m arm.

    p = inputParser;
    addParameter(p, 'root', fullfile(mcws_paths().single_channel, 'pt 11 emu 17 ch 328'));   % folders: mcws_paths.m
    addParameter(p, 'python_root', fullfile(mcws_paths().data, 'python_p11EMU17CH'));
    addParameter(p, 'channel', 328);
    addParameter(p, 'exp_type', 'RSVP_online');
    addParameter(p, 'arm_name', 'python_rescue_test');
    addParameter(p, 'python_times_file', '');
    addParameter(p, 'circshiftblanks', false);
    parse(p, varargin{:});
    o = p.Results;

    NSx = getfield(load(fullfile(o.root, 'NSx.mat')), 'NSx');
    pos = find(arrayfun(@(x) x.chan_ID == o.channel, NSx), 1);
    assert(~isempty(pos), 'build_python_arm_from_times: channel %d not found in NSx.mat', o.channel);
    nm = NSx(pos).output_name;

    % --- Resolve the Python times file (newest times_* subfolder, unless given) ---
    if ~isempty(o.python_times_file)
        py_times_path = o.python_times_file;
        assert(exist(py_times_path, 'file') == 2, 'python_times_file not found: %s', py_times_path);
    else
        d = dir(fullfile(o.python_root, 'times_*'));
        d = d([d.isdir]);
        assert(~isempty(d), 'build_python_arm_from_times: no times_* subfolders under %s', o.python_root);
        [~, order] = sort([d.datenum], 'descend');
        py_times_path = '';
        for k = order
            cand = fullfile(o.python_root, d(k).name, sprintf('times_%s.mat', nm));
            if exist(cand, 'file')
                py_times_path = cand;
                break;
            end
        end
        assert(~isempty(py_times_path), ...
            'build_python_arm_from_times: no times_%s.mat in any times_* subfolder under %s', nm, o.python_root);
    end
    fprintf('[build_python_arm_from_times] source: %s\n', py_times_path);

    times_py   = load(py_times_path);
    times_out  = adapt_python_times_for_matlab(times_py);
    spikes_out = adapt_python_spikes_for_matlab(times_py);
    fprintf('[build_python_arm_from_times] cluster_class rows=%d, cluster_class_pre_rescue rows=%d, rescued=%d\n', ...
        size(times_out.cluster_class,1), size(times_out.cluster_class_pre_rescue,1), nnz(times_out.rescue_mask));

    D = fullfile(o.root, o.arm_name);
    if ~exist(D, 'dir'), mkdir(D); end
    times_file  = fullfile(D, sprintf('times_%s.mat', nm));
    spikes_file = fullfile(D, sprintf('%s_spikes.mat', nm));
    save(times_file,  '-struct', 'times_out');
    save(spikes_file, '-struct', 'spikes_out');

    % templates_wc_offline.mat: on MATLAB arms cls_centers is exactly the
    % PRE-rescue class mean (verified 0 difference on rescue_sd1/sd2), so build
    % Python's the same way. spikes_out.spikes / cluster_class_pre_rescue are
    % the same rows in the same order (both = real class & ~rescued).
    pre_cls = times_out.cluster_class_pre_rescue(:,1);
    K = max([0; pre_cls]);
    % Only cls_centers/channels (what load_stored_template reads) -- MATLAB's
    % cls_maxdist uses a different definition than sqrt(sum(var)), so it's
    % left out rather than written under the same name with another meaning.
    centers = nan(K, size(spikes_out.spikes, 2));
    for kc = 1:K
        Xk = spikes_out.spikes(pre_cls == kc, :);
        if ~isempty(Xk), centers(kc,:) = mean(Xk, 1); end
    end
    cls_centers = {centers};
    channels = uint16(o.channel);
    save(fullfile(D, 'templates_wc_offline.mat'), 'cls_centers', 'channels');

    % SPC tree + labels (data_<nm>.dg_01 / .dg_01.lab) -- same name and ascii
    % format as MATLAB's, needed by temperature_diagram.m / the metrics report.
    py_dir = fileparts(py_times_path);
    for ext = {'.dg_01', '.dg_01.lab'}
        src = fullfile(py_dir, ['data_' nm ext{1}]);
        if exist(src, 'file')
            copyfile(src, fullfile(D, ['data_' nm ext{1}]));
        else
            fprintf('[build_python_arm_from_times] no %s in %s -- temperature diagram will be skipped\n', ['data_' nm ext{1}], py_dir);
        end
    end
    fprintf('[build_python_arm_from_times] wrote adapted times/spikes into %s\n', D);

    % do_structure_mu/sorted LOAD any grapes file already in D and add to it,
    % so rebuilding a folder must start from none (a rebuild stacked a new
    % run's mu raster onto the previous run's).
    old_grapes = dir(fullfile(D, 'grapes_blanks*.mat'));
    for gi = 1:numel(old_grapes)
        delete(fullfile(D, old_grapes(gi).name));
    end

    cd(o.root);
    do_structure_mu_BCM_online3(o.channel, o.exp_type, true, o.circshiftblanks, false, 'spike_dir', D);
    do_structure_sorted_BCM_online3(o.channel, true, o.circshiftblanks, false, 'times_dir', D, 'spike_dir', D);
    do_structure_rescue_BCM_online3(o.channel, true, o.circshiftblanks, false, 'times_dir', D, 'spike_dir', D);

    gf = fullfile(D, 'grapes_blanks_rescue.mat');
    assert(exist(gf, 'file') == 2, 'build_python_arm_from_times: expected %s, not written', gf);

    arm = struct('name', o.arm_name, 'dir', D, 'grapes_file', gf, ...
        'times_file', times_file, 'spikes_file', spikes_file);
    fprintf('[build_python_arm_from_times] done: %s\n', gf);
end
