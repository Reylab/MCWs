function [summary, pct_last_vs_prev] = build_arm_summary_table(arm_names, class_ids, n_in, mean_zscore, mu_base, std_base, out_dir, csv_name)
% BUILD_ARM_SUMMARY_TABLE  Builds the arm-vs-arm summary table (counts, %%
% diff vs the first/reference arm, z-score, baseline) and writes it to csv.
% Generalizes the two near-identical summary-table blocks that used to be
% hand-duplicated in test_rescue_grape_metrics.m (once for the stim_list-
% scoped "dynamic" summary, again for the all-stim "overall" summary) to an
% arbitrary number of arms.
%
% VARIABLES
%   arm_names/class_ids/n_in/mean_zscore/mu_base/std_base - inputs, one entry per arm,
%       same order; arm_names{1}/class_ids(1) is treated as the reference (e.g. GT) arm
%   out_dir/csv_name - where to write the csv
%   summary          - output: table with arm, class_id, n_in, pct_in_vs_gt, mean_zscore, mu_base, std_base
%   pct_last_vs_prev - output: %% diff between the LAST TWO arms (generalizes the one-off
%                      sd1->sd2 comparison to whichever two arms come last), NaN if <2 arms
    arm = cellstr(arm_names(:));
    class_id = class_ids(:);
    n_in = n_in(:); mean_zscore = mean_zscore(:); mu_base = mu_base(:); std_base = std_base(:);

    pct_in_vs_gt = 100*(n_in - n_in(1)) ./ n_in(1);
    pct_in_vs_gt(1) = NaN;

    pct_last_vs_prev = NaN;
    if numel(n_in) >= 2
        pct_last_vs_prev = 100*(n_in(end) - n_in(end-1)) / n_in(end-1);
    end

    summary = table(arm, class_id, n_in, pct_in_vs_gt, mean_zscore, mu_base, std_base, ...
                     'VariableNames', {'arm','class_id','n_in','pct_in_vs_gt','mean_zscore','mu_base','std_base'});
    disp(summary);
    if ~exist(out_dir,'dir'), mkdir(out_dir); end
    writetable(summary, fullfile(out_dir, csv_name));
end
