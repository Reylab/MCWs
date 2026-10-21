function out = mcws_validation(varargin)
% MCWS_VALIDATION  Every step of the MCWs (python) vs wave_clus vs GT validation for ONE channel folder:
% convert a python run, check grapes, compare, validate, build the for ppt report, the figures and the deck.
% See README_mcws_validation.md for the folder layout.
%
%   cd('<channel folder>')
%   mcws_validation('gt_class', 3, 'stims', [615 44 842])
%   mcws_validation('gt_class', [3 4], 'stims', {[133 418 404], [736 839 744]}, 'python_run', '<python output folder>')
%
% Name-value (only gt_class and stims are required):
%   'gt_class'    GT class(es) in hand selected/times_<name>.mat to report on (several = one report per class)
%   'stims'       stimulus numbers for the raster pages: a vector, or one vector per GT class (cell)
%   'folder'      the channel folder (default: the current folder)
%   'ch'          channel number (default: from <name>.NC5, '..._raw_<ch>.NC5' / '... raw_<ch>.NC5')
%   'python_run'  a MicroWireSorter output folder (test_full_pipeline.py) to convert first; '' = arms already there
%   'tag'         '' (arms in the channel folder, report in 'for ppt') or a run tag, e.g. 'isi3' -> arms in
%                 'run isi3', validation in 'validation isi3', report in 'for ppt isi3' (a tag starting with a
%                 digit, e.g. '10_19', gives 'python second pass 10_19', '10_19 validation', '10_19 for ppt')
%   'steps'       any of {'convert','report','figures','deck'} (default: all; 'convert' only with python_run)
%   'python'      python with MicroWireSorter's packages, for the figures (default <MCWS_ROOT>/.venv, mcws_paths)
%   'deck_python' python with python-pptx, for the deck (default ~/.venvs/pptdeck)
%   'rescue_sd'   the run's second-pass rescue radius, for the Euclidean / Mahalanobis figures (default 1)
%
% Output: struct with the report folder(s) and the PASS / FAIL of every unit's check.
p = inputParser;
addParameter(p, 'gt_class', []); addParameter(p, 'stims', []); addParameter(p, 'folder', pwd);
addParameter(p, 'ch', []); addParameter(p, 'python_run', ''); addParameter(p, 'tag', '');
addParameter(p, 'steps', {'convert', 'report', 'figures', 'deck'});
addParameter(p, 'python', fullfile(mcws_paths().mcws, '.venv', 'bin', 'python'));
addParameter(p, 'deck_python', fullfile(getenv('HOME'), '.venvs', 'pptdeck', 'bin', 'python'));
addParameter(p, 'rescue_sd', 1);
parse(p, varargin{:}); o = p.Results;
assert(~isempty(o.gt_class) && ~isempty(o.stims), 'mcws_validation: gt_class and stims are required');
here = fileparts(mfilename('fullpath'));
addpath(here, fullfile(fileparts(here), 'real_response_analysis'), fullfile(fileparts(here), 'responses'));
addpath(genpath(fullfile(fileparts(here), 'codes_emu')), genpath(fullfile(fileparts(here), 'useful_functions')));   % grapes builders
C = char(java.io.File(o.folder).getCanonicalPath());           % absolute channel folder
steps = cellstr(o.steps);

%% 0 what the folder must have
nc5 = dir(fullfile(C, '*.NC5')); assert(numel(nc5) == 1, 'mcws_validation: need exactly one .NC5 in %s', C);
[~, name] = fileparts(nc5.name); tf = ['times_' name '.mat'];
if isempty(o.ch), o.ch = str2double(regexp(name, '(\d+)$', 'tokens', 'once')); end
need = {'NSx.mat', 'finalevents.mat', 'stimulus.mat', 'experiment_properties_online3.mat', ...
        fullfile('hand selected', tf), fullfile('hand selected', 'grapes_blanks.mat'), fullfile('wave_clus notch', tf)};
miss = need(~cellfun(@(f) exist(fullfile(C, f), 'file') == 2, need));
assert(isempty(miss), 'mcws_validation: missing in %s: %s', C, strjoin(miss, ', '));
if isempty(o.tag), sub = ''; elseif isletter(o.tag(1)), sub = ['run ' o.tag]; else, sub = ['python second pass ' o.tag]; end
fprintf('[mcws_validation] %s | ch %d | GT class %s | arms in %s\n', C, o.ch, mat2str(o.gt_class), ternary(isempty(sub), '(channel folder)', sub));

%% 1 convert the python run (clustered, rescued, second pass) + grapes = times + metrics / second-pass report copies
if ~isempty(o.python_run) && any(strcmp(steps, 'convert'))
    convert_run(C, o.python_run, sub, o.ch, name);
end
assert(check_arm_grapes(C, sub, o.ch), 'mcws_validation: grapes do not match times in an arm (see [grapes] lines)');

%% 2-4 per GT class: comparisons + validation + for ppt report + checks
multi = numel(o.gt_class) > 1; out = struct('gt_class', {}, 'report', {}, 'check', {});
for j = 1:numel(o.gt_class)
    g = o.gt_class(j); st = o.stims; if iscell(st), st = st{j}; end
    id = validation_sessions('add', sprintf('local_ch%d_c%d', o.ch, g), C, o.ch, st, tf, g);
    if ~isempty(o.tag), id = [id '@' o.tag]; end
    pp = ppt_paths(validation_sessions(id), multi);
    if any(strcmp(steps, 'report'))
        ppt_session(id, multi);                 % wave_clus grapes, venns + class matches, the report (01-09)
        ppt_wc_py_destinations(id, multi);      % where every region's spikes are in the other sort
        ppt_mahal_forced_rescued(id, multi);    % 10_mahal_forced_rescued
        ppt_not_rescued_by_mask(id, multi);     % never-rescued quarantined spikes by mask + quarantine pages
        ok = ppt_check_unit(id, multi);         % accounting.csv
        ppt_flags(id, multi);                   % flags.csv
        second_pass_pages(C, pp);
    else
        ok = NaN;
    end
    M = load(fullfile(pp.rep, sprintf('membership_ch%d.mat', o.ch)), 'classes');
    if any(strcmp(steps, 'figures'))           % Euclidean vs Mahalanobis choice of the matched python class
        pyrun(o, sprintf('mahal_exact.py "%s" %d "%s" "%s"', C, M.classes.python, pp.rep, sub));
    end
    if any(strcmp(steps, 'deck'))
        system(sprintf('"%s" "%s" "%s"', o.deck_python, fullfile(here, 'build_channel_deck.py'), pp.rep));
    end
    out(end + 1) = struct('gt_class', g, 'report', pp.rep, 'check', ok); %#ok<AGROW>
end

%% 5 channel-level figures: reassignment distances / score / covariances
if any(strcmp(steps, 'figures'))
    rep0 = ppt_paths(validation_sessions(id), false).rep;
    pyrun(o, sprintf('reassign_scores.py "%s" "%s" "%s"', C, sub, fullfile(rep0, 'reassignment_scores')));
end
fprintf('[mcws_validation] done: %s\n', strjoin(arrayfun(@(u) sprintf('GT c%d %s', u.gt_class, ...
    ternary(isequal(u.check, true), 'PASS', ternary(isequal(u.check, false), 'FAIL', '-'))), out, 'UniformOutput', false), ', '));
end

function convert_run(C, run, sub, ch, name)
% the run's two top-level sorts (clustered: no quarantine.rescued; rescued) and its second pass -> python arms
g = dir(fullfile(C, 'hand selected', 'grapes*.mat'));
exp_type = getfield(load(fullfile(g(1).folder, g(1).name), 'exp_type'), 'exp_type'); %#ok<GFLD>
tf = [dir(fullfile(run, 'times_*', ['times_' name '.mat'])); dir(fullfile(run, 'second_pass_*', '*', 'times_*', ['times_' name '.mat']))];
assert(~isempty(tf), 'mcws_validation: no times_%s.mat in %s', name, run);
for f = tf'
    d = regexp(f.folder, 'times_(\d{4})-(\d{2})-(\d{2})', 'tokens'); d = d{end};
    if contains(f.folder, 'second_pass_'), kind = 'second_pass';
    else, Q = load(fullfile(f.folder, f.name), 'quarantine'); kind = ternary(isfield(Q.quarantine, 'rescued'), 'rescued', 'clustered'); end
    arm = fullfile(sub, sprintf('python_recluster_%s%s%s_%s', d{:}, kind));
    fprintf('[mcws_validation] convert %s -> %s\n', fullfile(f.folder, f.name), arm);
    old = pwd; build_python_arm_from_times('root', C, 'channel', ch, 'exp_type', exp_type, 'arm_name', arm, ...
        'python_times_file', fullfile(f.folder, f.name)); cd(old);
    if exist(fullfile(f.folder, 'metrics'), 'dir'), copyfile(fullfile(f.folder, 'metrics'), fullfile(C, arm, 'metrics')); end
    if strcmp(kind, 'second_pass')                         % second-pass report + where it came from
        sp = fileparts(f.folder); r = fullfile(C, arm, 'second_pass_report'); if ~exist(r, 'dir'), mkdir(r); end
        for x = {'second_pass_report.pdf', 'before_after.pdf', 'decisions.json'}
            if exist(fullfile(sp, x{1}), 'file'), copyfile(fullfile(sp, x{1}), r); end
        end
        if exist(fullfile(fileparts(sp), 'summary.csv'), 'file'), copyfile(fullfile(fileparts(sp), 'summary.csv'), r); end
        fid = fopen(fullfile(r, 'SOURCE.txt'), 'w'); fprintf(fid, '%s\n', sp); fclose(fid);
    end
end
end

function second_pass_pages(C, pp)
r = fullfile(C, fileparts(pp.py), pp.pyname, 'second_pass_report'); o = fullfile(pp.rep, '01_metrics', 'python_second_pass');
if ~exist(fullfile(r, 'second_pass_report.pdf'), 'file'), return; end
if ~exist(o, 'dir'), mkdir(o); end
system(sprintf('env -u LD_LIBRARY_PATH pdftoppm -r 110 -png "%s" "%s"', fullfile(r, 'second_pass_report.pdf'), fullfile(o, 'second_pass_report')));
if exist(fullfile(r, 'before_after.pdf'), 'file')
    system(sprintf('env -u LD_LIBRARY_PATH pdftoppm -r 110 -png "%s" "%s"', fullfile(r, 'before_after.pdf'), fullfile(o, 'before_after')));
end
end

function pyrun(o, args)
here = fileparts(mfilename('fullpath'));
s = system(sprintf('MCWS_RESCUE_SD=%g "%s" %s/%s', o.rescue_sd, o.python, here, args));
if s, warning('mcws_validation: python step failed: %s', args); end
end

function s = ternary(c, a, b)
if c, s = a; else, s = b; end
end
