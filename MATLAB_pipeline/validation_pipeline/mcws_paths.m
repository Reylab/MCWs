function P = mcws_paths()
% MCWS_PATHS  The folders the validation code uses, in one place. Each one can be set with an environment variable
% (before starting MATLAB, or with setenv inside it); otherwise the lab-server default is used.
%
%   P.data            MCWS_DATA_ROOT       data root                          default /mnt/data0
%   P.single_channel  MCWS_SINGLE_CHANNEL  one folder per channel             default <data>/MCWs Sessions/matlab validation/single channel
%   P.seeg_data       MCWS_SEEG_DATA       sEEG recordings (read only)        default <data>/sEEG_DATA
%   P.mcws            MCWS_ROOT            the MicroWireSorter repository     default /home/ajb/Documents/GitHub/MicroWireSorter
%   P.wave_clus       WAVE_CLUS_ROOT       wave_clus (wave_clus-master)       default /home/ajb/Documents/wave_clus-master
%   P.repo            (this repository's MATLAB_pipeline folder, from this file's location)
%
% The python and shell scripts read MCWS_ROOT the same way.
P.data           = env('MCWS_DATA_ROOT', '/mnt/data0');
P.single_channel = env('MCWS_SINGLE_CHANNEL', fullfile(P.data, 'MCWs Sessions', 'matlab validation', 'single channel'));
P.seeg_data      = env('MCWS_SEEG_DATA', fullfile(P.data, 'sEEG_DATA'));
P.mcws           = env('MCWS_ROOT', '/home/ajb/Documents/GitHub/MicroWireSorter');
P.wave_clus      = env('WAVE_CLUS_ROOT', '/home/ajb/Documents/wave_clus-master');
P.repo           = fileparts(fileparts(mfilename('fullpath')));
end

function v = env(name, default)
v = getenv(name); if isempty(v), v = default; end
end
