function st = load_validation_state(d)
% LOAD_VALIDATION_STATE  Load validation_state.mat written by run_validation,
% from D or the nearest folder above it (so it works from inside any of the
% output subfolders). D defaults to pwd.
%
% st.cfg         - the cfg the run used (st.cfg.out_dir = the folder found)
% st.arm_classes - class per arm the run used
% st.ref_tables  - per-stimulus data_table per unit, GT first then each arm
%                  (win_lo/win_hi = the raster windows)
% st.unit_names  - names for ref_tables ({gt, arm1, ...})
% st.onset_map   - stim number -> trial onsets (ms)
% st.Nscr        - sub-screening count

if nargin < 1 || isempty(d), d = pwd; end
d = char(d);
start = d;
while ~exist(fullfile(d, 'validation_state.mat'), 'file')
    up = fileparts(d);
    if strcmp(up, d)
        error('load_validation_state: no validation_state.mat in %s or above it -- run run_validation first.', start);
    end
    d = up;
end
st = load(fullfile(d, 'validation_state.mat'));
st.cfg.out_dir = d;    % the folder may have moved since the run
st.state_dir = d;
end
