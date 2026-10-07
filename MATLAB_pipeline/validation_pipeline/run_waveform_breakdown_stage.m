function run_waveform_breakdown_stage(cfg, arm_classes)
% RUN_WAVEFORM_BREAKDOWN_STAGE  Waveform breakdown pages per arm:
%   class<c>_rescued_by_property - the desired class c (cfg.arms(k).class,
%                         else the class matched to GT): the spikes rescued
%                         INTO it, one panel per quarantine property (the main
%                         per-class figure)
%   rescued_by_property - every rescued spike on the channel, one panel per
%                         quarantine property it had failed
%   not_rescued_by_property - quarantined spikes never rescued, one panel per
%                         quarantine property
% Each spike is in exactly one property panel: the one mask it failed, or
% 'multiple masks' if it failed 2+, so the panels add up to the page's n.
% Every panel has
% the matched (GT & MCWs) template and the desired class's sd1 template (II)
% (the legend names it). One fixed y range shared with every other validation
% waveform figure (prepare_arm_groups / wave_ylim).
%
% OUTPUTS (under cfg.out_dir/waveform_breakdowns)
%   <arm>_class<c>_rescued_by_property.png
%   <arm>_rescued_by_property.png / _not_rescued_by_property.png

if nargin < 2, arm_classes = []; end
[Gs, yl, ~, arm_classes] = prepare_arm_groups(cfg, arm_classes);
out_dir = fullfile(cfg.out_dir, 'waveform_breakdowns');

for k = 1:numel(cfg.arms)
    G = Gs{k};
    nm = cfg.arms(k).name;
    head = sprintf('ch%d  %s', cfg.ch, nm);
    for pc = G.per_class
        n_c = nnz(G.masks.resc & G.lab == pc.class);
        tag = '';
        if pc.class == arm_classes(k), tag = ', matched to GT'; end
        plot_wave_page(pc.resc_by_prop, pc.templates, yl, ...
            sprintf('%s -- rescued into class %d%s, by quarantine property (n=%d)', head, pc.class, tag, n_c), ...
            fullfile(out_dir, sprintf('%s_class%d_rescued_by_property.png', nm, pc.class)));
    end
    plot_wave_page(G.resc_by_prop, G.templates, yl, ...
        sprintf('%s -- all rescued spikes (every class) by quarantine property (n=%d)', head, nnz(G.masks.resc)), ...
        fullfile(out_dir, sprintf('%s_rescued_by_property.png', nm)));
    plot_wave_page(G.qnr_by_prop, G.templates, yl, ...
        sprintf('%s -- quarantined, never rescued, by quarantine property (n=%d)', head, nnz(G.masks.is_qnr)), ...
        fullfile(out_dir, sprintf('%s_not_rescued_by_property.png', nm)));
end
fprintf('[waveform breakdowns] y range fixed at [%.1f %.1f] -> %s\n', yl, out_dir);
end
