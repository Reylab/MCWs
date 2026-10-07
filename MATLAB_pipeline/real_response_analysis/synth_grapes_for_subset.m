function gf = synth_grapes_for_subset(abs_times, ch, root, template_grapes_file, out_dir, nm, circshiftblanks)
% SYNTH_GRAPES_FOR_SUBSET  Build a grapes_blanks.mat for a derived spike-time
% set that exists nowhere in any real grapes file (e.g. GT-not-SD1, an
% exclusive membership set) by calling the REAL update_grapes_blanks -- with
% domu=true, its own intended "just build mu" mode, not a workaround. A
% derived subset is a single undifferentiated spike-time list, not a
% clustering result -- there is no "class" to assign it to, so this fills
% grapes.rasters.chanN.mu (and .details.mu) via the real production
% windowing code, not a hand-rolled reimplementation.
%
% Copied verbatim from responses/analyze_cluster_membership.m (same name,
% same behavior) so it's callable as its own file.
%
% Returns gf = '' (no file written, nothing plotted) when abs_times is
% empty -- a genuinely empty overlap set is a real, valid result, not an
% error; callers must check for '' and skip plotting rather than crash.
    gf = '';
    abs_times = sort(abs_times(:)');
    if isempty(abs_times), return; end
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end
    T = load(template_grapes_file, 'ImageNames', 'exp_type');
    S = load(fullfile(root, 'stimulus.mat'));    stimulus = S.stimulus;
    F = load(fullfile(root, 'finalevents.mat')); pics_onset = F.pics_onset;
    E = load(fullfile(root, 'experiment_properties_online3.mat'), 'scr_config_cell', 'scr_end_cell');
    Nscr = min(numel(stimulus), numel(E.scr_end_cell));

    grapes = struct('time_pre', 1000, 'time_pos', 2000, 'exp_type', T.exp_type, 'ImageNames', {T.ImageNames});
    for scri = 1:Nscr
        grapes = update_grapes_blanks(grapes, pics_onset{scri}, F.seq_beg_blanks_cell{scri}, ...
            stimulus{scri}, {abs_times}, ch, {nm}, 1, {[]}, ...
            E.scr_config_cell{scri}.pics2use, scri, circshiftblanks, false);
    end
    gf = fullfile(out_dir, 'grapes_blanks.mat');
    save(gf, '-struct', 'grapes');
end
