function P = ppt_paths(cfg, multi)
% PPT_PATHS  Folders of one session's python arm, validation runs and for ppt report (cfg = validation_sessions(id)).
%   no run tag (id 'pt6emu13_ch264_class3'):
%     python arm  = newest python_recluster_*_second_pass in the channel folder
%     val         = the channel folder              (validation_gt_vs_<python>_class<g>, venn_gt_vs_python_class<g>)
%     wcval       = <channel>/wave_clus notch        (validation_wave_clus_vs_python_c<X>, venn_*, validation_gt_vs_wave_clus_notch_class<g>)
%     rep         = <channel>/for ppt[/GT c<g>]
%   run tag (id 'pt6emu13_ch264_class3@10_19'):
%     python arm  = newest python_recluster_*_second_pass in '<channel>/python second pass 10_19'
%     val         = '<channel>/10_19 validation',  wcval = '<channel>/10_19 validation/wave_clus notch'
%     rep         = '<channel>/10_19 for ppt[/GT c<g>]'
%   run tag starting with a letter (id 'pt6emu13_ch264_class3@isi3_sd1'): arm in '<channel>/run isi3_sd1',
%     val = '<channel>/validation isi3_sd1', wcval = its 'wave_clus notch', rep = '<channel>/for ppt isi3_sd1[/GT c<g>]'
%   The wave_clus run itself (times, grapes) is always <channel>/wave_clus notch (P.wc).
% P.py = python arm folder relative to the channel folder (fullfile(root, P.py) is the arm), P.pyname = its name.
if nargin < 2, multi = false; end
root = cfg.root; P.wc = fullfile(root, 'wave_clus notch');
tag = ''; if isfield(cfg, 'tag'), tag = cfg.tag; end
if isempty(tag)
    sub = ''; P.val = root; P.wcval = P.wc; P.rep = fullfile(root, 'for ppt');
elseif isletter(tag(1))                                  % e.g. isi3_sd1: run / validation / for ppt <tag>
    sub = ['run ' tag]; P.val = fullfile(root, ['validation ' tag]);
    P.wcval = fullfile(P.val, 'wave_clus notch'); P.rep = fullfile(root, ['for ppt ' tag]);
else                                                    % e.g. 10_19: python second pass <tag>, <tag> validation / for ppt
    sub = ['python second pass ' tag]; P.val = fullfile(root, [tag ' validation']);
    P.wcval = fullfile(P.val, 'wave_clus notch'); P.rep = fullfile(root, [tag ' for ppt']);
end
d = dir(fullfile(root, sub, 'python_recluster_*_second_pass'));
assert(~isempty(d), 'ppt_paths: no python_recluster_*_second_pass in %s', fullfile(root, sub));
P.pyname = d(end).name; P.py = fullfile(sub, P.pyname);
if multi, P.rep = fullfile(P.rep, sprintf('GT c%d', cfg.gt.class)); end
end
