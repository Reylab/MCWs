function result = mcws_validation(gt_class, stims, varargin)
% MCWS_VALIDATION  The MCWs vs wave_clus vs GT validation of one channel folder, in MATLAB:
%   1  check the channel folder has what the validation needs
%   2  (optional) convert a python sort into MATLAB arms
%   3  check every arm's grapes match its times file
%   4  for each GT unit: comparisons, validation, for ppt report, checks
%   5  (optional, only with 'python') the python figures, second-pass pages and slide deck
% Sections 1-4 need MATLAB only. The python sorter itself is run separately (README.md).
%
%   cd('<channel folder>')
%   mcws_validation(3, [615 44 842])                                     % arms already in the folder
%   mcws_validation([3 4], {[133 418 404], [736 839 744]}, ...
%       'python_run', '<python output folder>', 'tag', 'isi3')           % convert a python run first
%
% gt_class  GT class(es) in hand selected/times_<name>.mat
% stims     stimuli for the raster pages: a vector, or one vector per GT class in a cell
% options   'folder'      channel folder (default: current folder)
%           'python_run'  python output folder (test_full_pipeline.py) to convert first (default: none)
%           'tag'         keep this run apart: '' = arms in the channel folder, report in 'for ppt';
%                         'isi3' = arms in 'run isi3', report in 'for ppt isi3' (see ppt_paths.m)
%           'python_class' compare this python class instead of the one matched to GT (default [] = matched);
%                         one per GT class; the report then goes in '<report>/MCWs c<k>/'
%           'python'      python with MicroWireSorter's packages: also run section 5 (default '' = skip it)
%           'deck_python' python with python-pptx for the deck (default: 'python')
%           'rescue_sd'   the run's second-pass rescue radius, for the figures (default 1)
% result    one row per GT class: report folder and check (true = PASS)

opt.folder = pwd; opt.python_run = ''; opt.tag = ''; opt.python_class = []; opt.python = ''; opt.deck_python = ''; opt.rescue_sd = 1;
for k = 1:2:numel(varargin), opt.(varargin{k}) = varargin{k + 1}; end
if ~iscell(stims), stims = repmat({stims}, 1, numel(gt_class)); end
setup_path();
C = char(java.io.File(opt.folder).getCanonicalPath());          % the channel folder, full path

%% 1  what the channel folder needs
nc5 = dir(fullfile(C, '*.NC5'));
assert(numel(nc5) == 1, 'need exactly one .NC5 file in %s', C);
name = nc5.name(1:end - 4);                                     % e.g. 'mRAMY08 raw_264'
ch = str2double(regexp(name, '(\d+)$', 'tokens', 'once'));       % channel number from the name
times_file = ['times_' name '.mat'];
needed = {'NSx.mat', 'finalevents.mat', 'stimulus.mat', 'experiment_properties_online3.mat', ...
          fullfile('hand selected', times_file), fullfile('hand selected', 'grapes_blanks.mat'), ...
          fullfile('wave_clus notch', times_file)};
for f = needed
    assert(isfile(fullfile(C, f{1})), 'missing %s in %s', f{1}, C);
end
arm_folder = arm_subfolder(opt.tag);                             % where the python arms are

%% 2  convert a python sort (optional)
if ~isempty(opt.python_run)
    convert_python_run(C, opt.python_run, arm_folder, ch, name);
end

%% 3  grapes = times in every arm
assert(check_arm_grapes(C, arm_folder, ch), 'grapes do not match times in an arm (see the [grapes] lines)');

%% 4  each GT unit: comparisons + validation + for ppt report + checks
several = numel(gt_class) > 1;                                    % several GT units -> report in 'GT c<g>/'
result = struct('gt_class', {}, 'report', {}, 'check', {}, 'python_class', {});
for j = 1:numel(gt_class)
    g = gt_class(j);
    py_class = [];                                                  % [] = the python class matched to GT
    if ~isempty(opt.python_class), py_class = opt.python_class(j); end
    session = validation_sessions('add', sprintf('ch%d_c%d_py%d', ch, g, py_class), C, ch, stims{j}, times_file, g, py_class);
    if ~isempty(opt.tag), session = [session '@' opt.tag]; end

    ppt_session(session, several);                 % venns + class matches, the report (sections 01-09)
    ppt_wc_py_destinations(session, several);      % where each region's spikes are in the other sort
    ppt_mahal_forced_rescued(session, several);    % 10_mahal_forced_rescued
    ppt_not_rescued_by_mask(session, several);     % quarantined, never rescued, by mask
    passed = ppt_check_unit(session, several);     % accounting.csv
    ppt_flags(session, several);                   % flags.csv

    report = ppt_paths(validation_sessions(session), several).rep;
    m = load(fullfile(report, sprintf('membership_ch%d.mat', ch)), 'classes');     % the matched python class
    result(end + 1) = struct('gt_class', g, 'report', report, 'check', passed, 'python_class', m.classes.python); %#ok<AGROW>
    fprintf('[mcws_validation] GT c%d: %s -> %s\n', g, pass_fail(passed), report);
end

%% 5  python figures, second-pass pages, slide deck (optional: only with 'python')
if ~isempty(opt.python)
    if isempty(opt.deck_python), opt.deck_python = opt.python; end
    code = fileparts(mfilename('fullpath'));
    setenv('MCWS_RESCUE_SD', num2str(opt.rescue_sd));                  % read by the two figure scripts
    sp = dir(fullfile(C, arm_folder, 'python_recluster_*_second_pass'));
    sp_report = fullfile(sp(end).folder, sp(end).name, 'second_pass_report');
    for r = result
        pages = fullfile(r.report, '01_metrics', 'python_second_pass');
        if ~isfolder(pages), mkdir(pages); end
        run_external(sprintf('pdftoppm -r 110 -png "%s" "%s"', fullfile(sp_report, 'second_pass_report.pdf'), fullfile(pages, 'second_pass_report')));
        run_external(sprintf('pdftoppm -r 110 -png "%s" "%s"', fullfile(sp_report, 'before_after.pdf'), fullfile(pages, 'before_after')));
        run_external(sprintf('"%s" "%s" "%s" %d "%s" "%s"', opt.python, fullfile(code, 'mahal_exact.py'), C, r.python_class, r.report, arm_folder));
        run_external(sprintf('"%s" "%s" "%s"', opt.deck_python, fullfile(code, 'build_channel_deck.py'), r.report));
    end
    report_root = ppt_paths(validation_sessions(session), false).rep;
    run_external(sprintf('"%s" "%s" "%s" "%s" "%s"', opt.python, fullfile(code, 'reassign_scores.py'), C, arm_folder, ...
        fullfile(report_root, 'reassignment_scores')));
end
end


%% ---------- helpers ----------

function setup_path()
code = fileparts(mfilename('fullpath')); repo = fileparts(code);
addpath(code, fullfile(repo, 'real_response_analysis'), fullfile(repo, 'responses'));
addpath(genpath(fullfile(repo, 'codes_emu')), genpath(fullfile(repo, 'useful_functions')));   % grapes builders
end

function sub = arm_subfolder(tag)
% same naming as ppt_paths.m
if isempty(tag),          sub = '';
elseif isletter(tag(1)),  sub = ['run ' tag];
else,                     sub = ['python second pass ' tag];
end
end

function convert_python_run(C, run, arm_folder, ch, name)
% a python run has three sorts: times_<ts>/ (clustering), times_<ts>/ (+ rescue) and
% second_pass_<ts>/<label>_<ch>/times_<ts>/ -> arms python_recluster_<date>_clustered / _rescued / _second_pass
g = dir(fullfile(C, 'hand selected', 'grapes*.mat'));
exp_type = load(fullfile(g(1).folder, g(1).name), 'exp_type'); exp_type = exp_type.exp_type;
sorts = [dir(fullfile(run, 'times_*', ['times_' name '.mat'])); ...
         dir(fullfile(run, 'second_pass_*', '*', 'times_*', ['times_' name '.mat']))];
assert(~isempty(sorts), 'no times_%s.mat in %s', name, run);
for s = sorts'
    src = fullfile(s.folder, s.name);
    date = regexp(s.folder, 'times_(\d{4})-(\d{2})-(\d{2})', 'tokens'); date = [date{end}{:}];
    if contains(s.folder, 'second_pass_')
        kind = 'second_pass';
    else
        q = load(src, 'quarantine');
        if isfield(q.quarantine, 'rescued'), kind = 'rescued'; else, kind = 'clustered'; end
    end
    arm = fullfile(arm_folder, sprintf('python_recluster_%s_%s', date, kind));
    fprintf('[mcws_validation] %s -> %s\n', src, arm);
    here = pwd;
    build_python_arm_from_times('root', C, 'channel', ch, 'exp_type', exp_type, 'arm_name', arm, 'python_times_file', src);
    cd(here);
    if isfolder(fullfile(s.folder, 'metrics'))                    % the python metrics report
        copyfile(fullfile(s.folder, 'metrics'), fullfile(C, arm, 'metrics'));
    end
    if strcmp(kind, 'second_pass')                                % the second-pass report + where it came from
        copy_second_pass_report(fileparts(s.folder), fullfile(C, arm, 'second_pass_report'));
    end
end
end

function copy_second_pass_report(sp, out)
if ~isfolder(out), mkdir(out); end
for f = {'second_pass_report.pdf', 'before_after.pdf', 'decisions.json', fullfile('..', 'summary.csv')}
    if isfile(fullfile(sp, f{1})), copyfile(fullfile(sp, f{1}), out); end
end
fid = fopen(fullfile(out, 'SOURCE.txt'), 'w'); fprintf(fid, '%s\n', sp); fclose(fid);
end

function run_external(cmd)
% one command outside MATLAB (section 5 only); a failure is a warning, not an error
status = system(['env -u LD_LIBRARY_PATH ' cmd]);       % MATLAB's own libraries clash with poppler / python
if status ~= 0, warning('mcws_validation: failed (%d): %s', status, cmd); end
end

function s = pass_fail(ok)
if ok, s = 'PASS'; else, s = 'FAIL'; end
end
