function source = make_arm_temperature_plot(arm_dir, nm, arm_name, out_png)
% MAKE_ARM_TEMPERATURE_PLOT  One arm's wave_clus-style temperature diagram
% into the comparison output (run_rescue_grape_analysis.m), drawn from the arm
% folder's own data_<nm>.dg_01 (responses/temperature_diagram.m). Skipped with
% a message when the folder has no .dg_01. The metrics report no longer saves
% a stand-alone temperature PNG (it is the bottom-left panel of page 1), so
% there is nothing to copy from there.
% Any out_png left by an earlier run is removed first, so a skipped arm never
% shows a stale plot.
%
% VARIABLES
%   arm_dir  - input: the arm's folder (times_<nm>.mat lives here)
%   nm       - input: channel output name, e.g. 'mLFUS08 raw_328'
%   arm_name - input: label used in the log lines
%   out_png  - input: where to write the plot
%   source   - output: 'dg' or 'none'
    source = 'none';
    if exist(out_png, 'file'), delete(out_png); end
    if ~isempty(dir(fullfile(arm_dir, 'data_*.dg_01')))
        try
            temperature_diagram(arm_dir, nm, [], out_png);
        catch ME
            fprintf('[temperature] %s: temperature_diagram failed (%s)\n', arm_name, ME.message);
        end
        if exist(out_png, 'file')
            fprintf('[temperature] %s: drawn from %s\n', arm_name, arm_dir);
            source = 'dg';
            return;
        end
    end
    fprintf('[temperature] %s: no data_*.dg_01 in %s -- skipped\n', arm_name, arm_dir);
end
