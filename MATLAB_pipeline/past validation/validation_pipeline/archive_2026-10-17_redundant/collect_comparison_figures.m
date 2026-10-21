function collect_comparison_figures(channel, wc_folder, out_root, varargin)
% COLLECT_COMPARISON_FIGURES  Copy one channel's comparison figures into a review folder (copies only,
% nothing is changed). Layout, under <out_root>/<chNNN>/:
%   wave_clus_spikes/          the wave_clus run's fig2print*.png
%   python_spikes/             the cluster pages of python's metrics report (report_*.pdf -> png)
%   gt_vs_python_class<g>/     class_match_*.png + venn_three_set/*_cluster.png
%   gt_vs_wave_clus_class<g>/  same, from <wc_folder>/venn_gt_vs_wave_clus_class<g>
%   wave_clus_vs_python_c<X>/  same, from <wc_folder>/venn_wave_clus_vs_python_c<X>
%   no_notch_vs_notch/         compare_detections pages (detections_*.png + the why-missing csv)
%
%   SC = '/mnt/data0/MCWs Sessions/matlab validation/single channel';
%   collect_comparison_figures(fullfile(SC, 'pt 6 emu 13 ch 262'), 'wave_clus no notch', fullfile(SC, 'temp_review'))
%
% Name-value:
%   'python'   python arm folder (default: newest python_recluster_*_second_pass)
%   'compare'  compare_detections output folder (default: <channel>/compare_no_notch_vs_notch, if present)
%   'label'    put the wave_clus figures (wave_clus_spikes, gt_vs_wave_clus, wave_clus_vs_python) in
%              <chNNN>/<label>/ -- e.g. 'notch' and 'no_notch' for two runs of one channel
%   'shared'   also copy the figures that do not depend on the wave_clus run: python_spikes,
%              gt_vs_python, no_notch_vs_notch (default true; false for a channel's second run)
p = inputParser;
addParameter(p, 'python', '');
addParameter(p, 'compare', '');
addParameter(p, 'label', '');
addParameter(p, 'shared', true);
parse(p, varargin{:}); o = p.Results;
ch = regexp(channel, 'ch (\d+)$', 'tokens', 'once'); out = fullfile(out_root, ['ch' ch{1}]);
outw = fullfile(out, o.label);
if ~exist(outw, 'dir'), mkdir(outw); end
wc = fullfile(channel, wc_folder);

cp(dir(fullfile(wc, 'fig2print*.png')), fullfile(outw, 'wave_clus_spikes'));

if isempty(o.python)
    d = dir(fullfile(channel, 'python_recluster_*_second_pass')); o.python = d(end).name;
end
rep = dir(fullfile(channel, o.python, 'metrics', 'report_*.pdf'));
if ~isempty(rep) && o.shared
    pdf = fullfile(rep(1).folder, rep(1).name); dst = fullfile(out, 'python_spikes'); mkdir_(dst);
    % MATLAB's own library path breaks poppler: run pdftotext / pdftoppm with it cleared
    [~, txt] = system(sprintf('LD_LIBRARY_PATH= pdftotext -f 1 -l 1 "%s" - | grep -o "Page 1/[0-9]*"', pdf));
    n = str2double(regexp(txt, '\d+$', 'match', 'once')); if isnan(n), n = 1; end
    system(sprintf('LD_LIBRARY_PATH= pdftoppm -r 110 -png -f 1 -l %d "%s" "%s"', n, pdf, fullfile(dst, [o.python '_cluster_page'])));
end

gtp = [dir(fullfile(channel, 'venn_gt_vs_python_class*')); dir(fullfile(channel, 'validation_gt_vs_python*'))];
if ~o.shared, gtp = gtp([]); end
venns = [gtp; dir(fullfile(wc, 'venn_gt_vs_wave_clus_class*')); dir(fullfile(wc, 'venn_wave_clus_vs_python_c*'))];
for v = venns'
    src = fullfile(v.folder, v.name);
    name = regexprep(v.name, '^venn_', '');
    dst = outw;
    if startsWith(v.name, 'validation_gt_vs_python'), name = 'gt_vs_python'; end
    if startsWith(name, 'gt_vs_python'), dst = out; end
    cp([dir(fullfile(src, 'class_match_*.png')); dir(fullfile(src, 'venn_three_set', '*_cluster.png'))], fullfile(dst, name));
end

if isempty(o.compare), o.compare = fullfile(channel, 'compare_no_notch_vs_notch'); end
if exist(o.compare, 'dir') && o.shared
    cp([dir(fullfile(o.compare, 'detections_*.png')); dir(fullfile(o.compare, 'detections_why_missing.csv'))], fullfile(out, 'no_notch_vs_notch'));
end
fprintf('[collect] %s -> %s\n', channel, out);
end

function cp(files, dst)
if isempty(files), return; end
mkdir_(dst);
for f = files', copyfile(fullfile(f.folder, f.name), dst); end
end

function mkdir_(d)
if ~exist(d, 'dir'), mkdir(d); end
end
