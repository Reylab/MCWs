function pdf = make_validation_report(cfgs, pdf)
% MAKE_VALIDATION_REPORT  One PDF of a run's pages (or several runs'), in
% reading order:
%   1. metrics   - class match per arm, GT-vs-ARM summary and sign-test pages
%   2. venn      - three-set venns (cluster, then full) per arm
%   3. pie       - cluster composition per arm
%   4. pie-venn  - pie groups by venn region per arm
%   5. rasters   - GT, each arm, then each arm's subset rasters, then the
%                  response-window waveforms
%   6. waveforms - waveform breakdowns per arm
%
%   make_validation_report(cfg)                 % -> <out_dir>/pdf_reports/validation_report.pdf
%   make_validation_report({cfg1, cfg2, ...}, '/path/all_channels.pdf')
%   (run_validation calls it last, stage 'report')
%
% Each page carries a caption: channel | section | file.

here = fileparts(mfilename('fullpath'));
addpath(fullfile(fileparts(here), 'real_response_analysis'));
if isstruct(cfgs), cfgs = {cfgs}; end
if nargin < 2 || isempty(pdf)
    pdf = fullfile(cfgs{1}.out_dir, 'pdf_reports', 'validation_report.pdf');
end

pngs = {}; caps = {};
for c = 1:numel(cfgs)
    [p, k] = report_pages(cfgs{c});
    pngs = [pngs, p]; caps = [caps, k]; %#ok<AGROW>
end
compile_pngs_to_pdf(pngs, pdf, caps);
fprintf('[report] %d pages -> %s\n', numel(pngs), pdf);
end

function [pngs, caps] = report_pages(cfg)
    o = cfg.out_dir;
    arms = {cfg.arms.name};
    [~, run] = fileparts(o);
    tag = sprintf('ch%d GT class %d (%s)', cfg.ch, cfg.gt.class, run);
    S = cell(0, 2);   % {section, files}

    % 1. metrics
    f = {};
    for a = arms
        f = [f, ls_png(o, sprintf('class_match_ch*_%s.png', a{1})), ...
                ls_png(o, sprintf('class_match_ch*_%s_vs_*.png', a{1})), ...
                ls_png(fullfile(o, 'stimulus_rasters', a{1}), '*gt_vs_arm_summary.png'), ...
                ls_png(fullfile(o, 'stimulus_rasters', a{1}), '*sign_test.png')]; %#ok<AGROW>
    end
    S(end+1, :) = {'metrics', f};
    % 2-4. venn, pie, pie-venn
    f = {};
    for a = arms
        f = [f, ls_png(fullfile(o, 'venn_three_set'), sprintf('venn3_*_%s_cluster.png', a{1})), ...
                ls_png(fullfile(o, 'venn_three_set'), sprintf('venn3_*_%s_full.png', a{1}))]; %#ok<AGROW>
    end
    S(end+1, :) = {'venn', f};
    S(end+1, :) = {'pie', cellfun(@(a) ls_png(fullfile(o, 'cluster_composition'), sprintf('composition_*_%s.png', a)), ...
        arms, 'UniformOutput', false)};
    S(end+1, :) = {'pie-venn', cellfun(@(a) ls_png(fullfile(o, 'pie_venn'), sprintf('pie_venn_*_%s.png', a)), ...
        arms, 'UniformOutput', false)};
    % 5. rasters: GT, arms, subsets (gt_only, arm_only, matched, quarantined, by property), response waveforms
    f = ls_png(fullfile(o, 'stimulus_rasters', cfg.gt.name), '*resp_window*.png');
    for a = arms
        f = [f, ls_png(fullfile(o, 'stimulus_rasters', a{1}), '*resp_window*.png')]; %#ok<AGROW>
    end
    order = {'gt_only', 'arm_only', 'matched', 'q_not_rescued'};
    for a = arms
        d = fullfile(o, 'subset_rasters', a{1});
        if ~exist(d, 'dir'), continue; end
        sub = dir(d); sub = {sub([sub.isdir] & ~startsWith({sub.name}, '.')).name};
        sub = [intersect(order, sub, 'stable'), sort(sub(startsWith(sub, 'qnr_'))), setdiff(sub, [order, sub(startsWith(sub, 'qnr_'))])];
        for s = sub
            f = [f, ls_png(fullfile(d, s{1}), '*resp_window*.png')]; %#ok<AGROW>
        end
    end
    for a = arms
        f = [f, ls_png(fullfile(o, 'response_waveforms'), sprintf('response_waveforms_*_%s.png', a{1}))]; %#ok<AGROW>
    end
    S(end+1, :) = {'rasters', f};
    % 6. waveform breakdowns
    f = {};
    for a = arms
        f = [f, ls_png(fullfile(o, 'waveform_breakdowns'), sprintf('%s_*.png', a{1}))]; %#ok<AGROW>
    end
    S(end+1, :) = {'waveforms', f};

    pngs = {}; caps = {};
    for i = 1:size(S, 1)
        files = S{i, 2};
        if iscell(files) && ~isempty(files) && iscell(files{1}), files = [files{:}]; end
        for j = 1:numel(files)
            [~, nm] = fileparts(files{j});
            pngs{end+1} = files{j}; %#ok<AGROW>
            caps{end+1} = sprintf('%s | %s | %s', tag, S{i, 1}, nm); %#ok<AGROW>
        end
    end
end

function f = ls_png(d, pat)
    e = dir(fullfile(d, pat));
    f = sort(fullfile({e.folder}, {e.name}));
    f = f(:)';
end
