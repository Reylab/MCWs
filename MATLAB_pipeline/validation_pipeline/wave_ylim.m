function yl = wave_ylim(Gs, min_n_band, min_n_mean)
% WAVE_YLIM  One fixed y range for every waveform panel of every validation
% figure, over every arm in Gs (cell of arm_wave_groups outputs), padded 8%:
%   - the mean +/- std band of every group with >= MIN_N_BAND spikes (default 50)
%   - the mean line of every group with >= MIN_N_MEAN spikes (default 5)
%   - the templates
% So every group's mean is always fully on the axis, and only the std band of
% a small noisy group (e.g. a handful of spikes rescued into a big unit) can
% run past it -- draw_wave_group flags those panels -- instead of a few odd
% spikes stretching the axis for every other panel.
    if nargin < 2, min_n_band = 50; end
    if nargin < 3, min_n_mean = 5; end
    lo = inf; hi = -inf;
    for k = 1:numel(Gs)
        G = Gs{k};
        groups = [G.comp, G.c0, G.qnr, G.resc_by_prop, G.qnr_by_prop, ...
                  [G.per_class.resc_by_prop]];
        for g = groups
            n = size(g.W, 1);
            if n < min_n_mean, continue; end
            mu = mean(g.W, 1);
            if n >= min_n_band
                sd = std(g.W, 0, 1);
                lo = min(lo, min(mu - sd)); hi = max(hi, max(mu + sd));
            else
                lo = min(lo, min(mu)); hi = max(hi, max(mu));
            end
        end
        for t = [G.templates(:)', G.per_class.templates]
            lo = min(lo, min(t.wave)); hi = max(hi, max(t.wave));
        end
    end
    yl = [lo hi] + 0.08*(hi - lo)*[-1 1];
end
