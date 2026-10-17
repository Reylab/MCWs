function plot_class_rasters(state, unit_dir, grapes_file, times_file, k, name, out_dir)
% PLOT_CLASS_RASTERS  Raster pages of one whole class with ITS OWN response windows (latency / duration from
% that class's own grapes), exactly as run_validation's rasters stage draws a unit: <out_dir>/<name>, plus
% <out_dir>/<name>_default_window when one of its stimuli has a late latency (> cfg.max_lat) or a long
% response (> cfg.max_dura) -- those stimuli then get grapes' default post window. Footer says which.
% STATE: any validation folder of the session (stim list, screening count, onset map, window settings).
%   plot_class_rasters(state, fullfile(root, 'wave_clus notch'), 'grapes_blanks.mat', tf, 2, 'wave_clus_c2', out)
st = load_validation_state(state); cfg = st.cfg; ch = cfg.ch;
if ~isfield(cfg, 'max_dura'), cfg.max_dura = 250; end
if ~isfield(cfg, 'max_lat'), cfg.max_lat = 400; end
m = get_class_metrics(unit_dir, grapes_file, ch, k, cfg.stim_list, cfg.root);
grapes = load(fullfile(unit_dir, grapes_file));
T = load(fullfile(unit_dir, times_file), 'cluster_class'); cc = T.cluster_class;
ps = m.per_stim; rc = m.resp_conf;
ps.post_lo = ps.tons; ps.post_hi = ps.tons + m.rank_config.win_length;
ps.base_lo = repmat(-rc.tmax_median, height(ps), 1); ps.base_hi = repmat(-rc.tmin_median, height(ps), 1);
fl = ps.dura > cfg.max_dura | ps.onset > cfg.max_lat; ids = ps.stim_number(fl);
vers = {'defined'}; if any(fl), vers{end + 1} = 'default'; end
t_k = cc(cc(:, 1) == k, 2);
for v = vers
    lo = ps.onset - cfg.dyn_buffer; hi = ps.onset + ps.dura + cfg.dyn_buffer; nm = name;
    if strcmp(v{1}, 'default')
        lo(fl) = rc.tmin_median; hi(fl) = rc.tmax_median; nm = [name '_default_window'];
        footer = sprintf(['own class windows: DEFAULT WINDOW %g-%g ms after onset for stim %s (latency > %d or ' ...
            'duration > %d ms); other stimuli: latency / duration'], rc.tmin_median, rc.tmax_median, mat2str(ids(:)'), cfg.max_lat, cfg.max_dura);
    else
        footer = sprintf('own class windows: latency / duration (onset-%d to onset+dura+%d ms)', cfg.dyn_buffer, cfg.dyn_buffer);
        if any(fl)
            footer = sprintf('%s | stim %s: latency > %d or duration > %d ms -> also a *_default_window version', ...
                footer, mat2str(ids(:)'), cfg.max_lat, cfg.max_dura);
        end
    end
    n_in = count_in_windows(t_k, ps.stim_number, lo, hi, st.onset_map);
    plot_response_window_grapes(grapes, ps, ch, k, n_in, lo, hi, m.rank_config, m.ejex, st.Nscr, nm, ...
        fullfile(out_dir, nm), cfg.root, [], false, numel(t_k), [], 'x_jitter', cfg.x_jitter, 'footer', footer);
    fprintf('[class rasters] %s: %d spikes, %d in window -> %s\n', nm, numel(t_k), n_in, fullfile(out_dir, nm));
end
end
