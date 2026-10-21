function subset_sets(unit_ids)
% SUBSET_SETS  Raster pages for the compact report's exclusive and other-class
% spike sets (subset_sets.py writes <out_dir>/compact/sets/<unit>/manifest.json):
% plot_subset_rasters with custom times, in the validation and with the windows
% of each set's own unit (manifest 'dir' / 'side').
%   subset_sets()                 % every unit with a manifest
%   subset_sets({'ch337_GT4'})
% -> <validation>/subset_rasters/python_sd1/<set>/
here = fileparts(mfilename('fullpath'));
addpath(fileparts(here));
C = jsondecode(fileread(fullfile(here, 'channels.json')));
D = dir(fullfile(C.out_dir, 'compact', 'sets', '*', 'manifest.json'));
for d = D'
    M = jsondecode(fileread(fullfile(d.folder, d.name)));
    if nargin && ~ismember(M.unit, cellstr(unit_ids)), continue; end
    for s = M.sets'
        done = dir(fullfile(s.dir, 'subset_rasters', 'python_sd1', s.name, '*.png'));
        if ~isempty(done), fprintf('[%s] %s: already drawn\n', M.unit, s.name); continue; end
        t = load(s.file, 'times'); t = t.times;
        if isempty(t), fprintf('[%s] %s: no spikes\n', M.unit, s.name); continue; end
        plot_subset_rasters([], 'dir', s.dir, 'times', t, 'name', s.name, 'side', s.side);
        fprintf('[%s] %s: %d spikes -> %s\n', M.unit, s.name, numel(t), s.dir);
    end
end
end
