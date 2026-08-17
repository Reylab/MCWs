function parse_NSx_sim(varargin)
% PARSE_NSX_SIM  Build an NSx.mat metadata file (+ per-channel raw binary
% files) from simulated single-channel .mat files, so that the existing
% pipeline (Get_spikes -> Do_features -> Do_clustering -> refract_viol ...)
% can run on simulated data WITHOUT any changes to those functions.
%
% This mirrors what parse_NSx.m produces for real Blackrock recordings:
%   - one raw int16 binary file per channel (same convention: readInData
%     reads output_name+ext as a raw binary stream)
%   - a metadata file 'NSx.mat' containing NSx (struct array) and files
%
% Expected input files: '<folder>/simulation_<N>.mat', each containing a
% single variable 'data' (Nx1 or 1xN double vector = one channel's trace).
% The trailing number <N> is used as chan_ID (same trick Get_spikes.m
% uses: regexp(f,'\d+$','match')).
%
% Optional name-value parameters:
%   'folder'       - folder containing simulation_*.mat files (default: pwd)
%   'pattern'      - regexp/glob-ish pattern to find files (default: 'simulation_*.mat')
%   'sr'           - sampling rate in Hz, used as default if a file doesn't
%                    specify its own 'sr' variable (default: 24000)
%   'is_micro'     - default is_micro flag if a file doesn't specify its own
%                    'is_micro' variable (default: true)
%   'unit'         - default unit string if a file doesn't specify 'unit'
%                    (default: 'uV'). NOTE: this is an assumption about your
%                    simulated data's units -- change if it's wrong.
%   'ext'          - file extension used for the raw binary files written to
%                    disk (default: '.NC5'). Must be an extension your
%                    readInData/supported_wc_extensions setup recognizes as
%                    a raw int16 stream. Change this if your pipeline expects
%                    something else.
%   'which_system' - tag stored in NSx(k).which_system (default: 'SIM')
%   'overwrite'    - if true, re-parse & overwrite channels that already
%                    exist in NSx.mat (default: false)
%   'genNC'        - if true, also write the raw int16 binary file per
%                    channel (needed by anything that reads continuous
%                    data off disk, e.g. new_check_lfp_power_NSX.m).
%                    If false, only NSx.mat metadata is produced (default: false)
%
% Bundle handling: since these channels are independent (no cross-channel
% artifact rejection needed), each channel is assigned its OWN unique
% bundle label. This means bundle_artifact.m would treat every channel as
% a singleton bundle and never flag anything as a collision artifact even
% if it were accidentally run -- but you said you won't be running it on
% this data anyway.
%
% Example (metadata only -- e.g. within_channel/refract_viol/Do_clustering,
% which only need chan_ID + output_name, not raw continuous data):
%   parse_NSx_sim('folder', '/data/sim_batch1', 'sr', 24000);
%   within_channel([1:95])
%   refract_viol([1:95])
%   Do_clustering('all')
%
% Example (also write raw .NC5 binaries -- needed for e.g.
% new_check_lfp_power_NSX.m, which reads continuous data off disk):
%   parse_NSx_sim('folder', '/data/sim_batch1', 'sr', 24000, 'genNC', true);

p = inputParser;
addParameter(p, 'folder', pwd, @ischar);
addParameter(p, 'pattern', 'simulation_*.mat', @ischar);
addParameter(p, 'sr', 24000, @isnumeric);
addParameter(p, 'is_micro', true, @islogical);
addParameter(p, 'unit', 'uV', @ischar);
addParameter(p, 'ext', '.NC5', @ischar);
addParameter(p, 'which_system', 'SIM', @ischar);
addParameter(p, 'overwrite', false, @islogical);
addParameter(p, 'genNC', false, @islogical);
parse(p, varargin{:});

folder       = p.Results.folder;
pattern      = p.Results.pattern;
default_sr   = p.Results.sr;
default_micro= p.Results.is_micro;
default_unit = p.Results.unit;
file_ext     = p.Results.ext;
which_system = p.Results.which_system;
overwrite    = p.Results.overwrite;
genNC        = p.Results.genNC;

d = dir(fullfile(folder, pattern));
if isempty(d)
    error('No files matching ''%s'' found in %s', pattern, folder);
end

% Load existing metadata (if any) 
metadata_file = fullfile(folder, 'NSx.mat');
if exist(metadata_file, 'file')
    metadata = load(metadata_file);
    NSx = metadata.NSx;
    files = metadata.files;
else
    NSx = struct([]);
    files = struct([]);
end

fprintf('Found %d simulated channel file(s) in %s\n', numel(d), folder);

for i = 1:numel(d)
    fname = d(i).name;
    full_path = fullfile(folder, fname);

    % --- chan_ID from trailing number in filename ---
    num_match = regexp(fname, '\d+', 'match');
    if isempty(num_match)
        warning('Could not extract a channel number from ''%s''. Skipping.', fname);
        continue
    end
    chan_ID = str2double(num_match{end});

    % --- skip/overwrite handling ---
    if ~isempty(NSx)
        existing = find(arrayfun(@(x) x.chan_ID==chan_ID, NSx));
    else
        existing = [];
    end
    if ~isempty(existing) && ~overwrite
        fprintf('Skipping channel %d, already parsed. (pass ''overwrite'',true to redo)\n', chan_ID);
        continue
    end

    % --- load the sim file and pull optional overrides if present ---
    S = load(full_path);
    if ~isfield(S, 'data')
        warning('%s has no variable ''data''. Skipping.', fname);
        continue
    end
    data = double(S.data(:))'; % force row vector, double precision

    sr       = getfield_or_default(S, 'sr',       default_sr);
    is_micro = getfield_or_default(S, 'is_micro', default_micro);
    unit_str = getfield_or_default(S, 'unit',     default_unit);
    label    = getfield_or_default(S, 'label',    sprintf('simulation_%d', chan_ID));
    bundle   = getfield_or_default(S, 'bundle',   num2str(chan_ID)); % unique per channel by default

    % --- scale to int16 so the raw binary file matches what readInData
    % expects (real int16 stream + a 'conversion' factor back to real units) ---
    if isfield(S, 'conversion')
        conversion = S.conversion;
    else
        max_abs = max(abs(data));
        if max_abs == 0 || isnan(max_abs)
            conversion = 1;
        else
            conversion = max_abs / 32767;
        end
    end

    % output_name matches the simulation_<N> naming your edited Get_spikes.m
    % already used to produce '<output_name>_spikes.mat' files.
    output_name = sprintf('simulation_%d', chan_ID);
    out_file = [output_name file_ext];

    % --- Raw binary file: only written when genNC is true. Needed by
    % anything reading continuous data off disk (e.g. new_check_lfp_power_NSX.m).
    % Not needed for within_channel.m / refract_viol.m / Do_clustering.m,
    % which only read existing '<output_name>_spikes.mat' files.
    if genNC
        scaled = int16(round(data / conversion));
        fid = fopen(fullfile(folder, out_file), 'w');
        if fid == -1
            error('Could not open %s for writing.', fullfile(folder, out_file));
        end
        fwrite(fid, scaled, 'int16');
        fclose(fid);
    end

    % --- populate NSx struct entry ---
    if isempty(existing)
        pos = numel(NSx) + 1;
    else
        pos = existing;
    end
    NSx(pos).chan_ID      = chan_ID;
    NSx(pos).electrode_ID = chan_ID;
    NSx(pos).output_name  = output_name;
    NSx(pos).label        = label;
    NSx(pos).macro        = output_name;
    NSx(pos).unit         = unit_str;
    NSx(pos).conversion   = conversion;
    NSx(pos).sr           = sr;
    NSx(pos).nsp          = [];
    NSx(pos).which_system = which_system;
    NSx(pos).ext          = file_ext;
    NSx(pos).lts          = numel(data);
    NSx(pos).filename     = {full_path};
    NSx(pos).is_micro     = is_micro;
    NSx(pos).bundle       = bundle;
    NSx(pos).dc           = 0;

    if genNC
        fprintf('Channel %d (%s): %d samples, sr=%d Hz, conversion=%.4g -> wrote %s\n', ...
            chan_ID, output_name, numel(data), sr, conversion, out_file);
    else
        fprintf('Channel %d (%s): %d samples, sr=%d Hz, conversion=%.4g (metadata only, no %s written)\n', ...
            chan_ID, output_name, numel(data), sr, conversion, out_file);
    end

    % --- track file provenance like parse_NSx.m does ---
    fpos = numel(files) + 1;
    if ~isempty(files)
        frep = arrayfun(@(x) strcmp(x.name, full_path), files);
        if any(frep)
            fpos = find(frep);
        end
    end
    files(fpos).name = full_path;
    files(fpos).first_sample = 1;
    files(fpos).lts = numel(data);
    files(fpos).which_nsp = [];
    files(fpos).trim4sinc = 0;
    files(fpos).which_cells = 1;
end

freq_priority = [30000, 2000, 10000, 1000, 500];
save(metadata_file, 'NSx', 'files', 'freq_priority');
fprintf('NSx.mat written to %s with %d channel(s).\n', metadata_file, numel(NSx));

end

function val = getfield_or_default(S, fname, default_val)
    if isfield(S, fname)
        val = S.(fname);
    else
        val = default_val;
    end
end