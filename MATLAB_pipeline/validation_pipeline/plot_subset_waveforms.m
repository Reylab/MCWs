function plot_subset_waveforms(groups, templates, ttl, out_png, varargin)
% PLOT_SUBSET_WAVEFORMS  One page of waveform panels: each spike subset as its single spikes (up to
% 5000, no std band) + mean, with the class templates of the sorts drawn on every panel.
%
%   g = struct('name', {'wave_clus c1 only', 'matched'}, 'W', {W1, W2}, 'color', {[0 .45 .7], [.35 .35 .35]});
%   t = templates_from(M, {'gt', 1; 'wave_clus', 1; 'python', 1});   % or build the struct yourself:
%   t = struct('name', {'GT c1'}, 'wave', {M.templates.gt(1, :)}, 'color', {[0 0 0]}, 'style', {'--'});
%   plot_subset_waveforms(g, t, 'ch259 ...', fullfile(out, 'waveforms.png'))
%
% groups    struct array: name, W (spikes x samples), color
% templates struct array: name, wave, color, style ('-' / '--' / ':')
% Name-value: 'ncol' panels per row (default 3), 'ylim' (default: from all groups and templates)
p = inputParser;
addParameter(p, 'ncol', 3);
addParameter(p, 'ylim', []);
parse(p, varargin{:}); o = p.Results;
addpath(fileparts(mfilename('fullpath')));                 % draw_wave_group, plot_traces
ng = numel(groups); nr = ceil(ng / o.ncol);
yl = o.ylim;
if isempty(yl)
    allw = [];
    for g = groups(:)', if ~isempty(g.W), allw = [allw; reshape(prctile(g.W(:), [0.5 99.5]), 1, 2)]; end, end %#ok<AGROW>
    for t = templates(:)', allw = [allw; min(t.wave) max(t.wave)]; end %#ok<AGROW>
    if isempty(allw), yl = [-100 50]; else, yl = [min(allw(:, 1)) max(allw(:, 2))]; yl = yl + 0.1 * diff(yl) * [-1 1]; end
end
f = figure('Visible', 'off', 'Position', [50 50 520 * o.ncol 430 * nr + 80], 'Color', 'w');
for i = 1:ng
    ax = subplot(nr, o.ncol, i, 'Parent', f);
    draw_wave_group(ax, groups(i), templates, yl, i == 1);
end
sgtitle(f, ttl, 'Interpreter', 'none', 'FontWeight', 'bold');
print(f, out_png, '-dpng', '-r110'); close(f);
end
