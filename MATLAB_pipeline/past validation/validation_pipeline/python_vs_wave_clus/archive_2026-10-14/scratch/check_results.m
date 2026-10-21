D = '/mnt/data0/MCWs Sessions/matlab validation/single channel/pt 10 emu 7 ch 368/validation_gt_vs_python_recluster_20261006_sd2/results';
P = readtable(fullfile(D, 'per_stimulus.csv'));
Tr = readtable(fullfile(D, 'per_trial.csv'));
U = readtable(fullfile(D, 'per_unit.csv'));
R = load(fullfile(D, 'rasters.mat'), 'spikes');
bad = zeros(1, 5);
for i = 1:height(P)
    t = Tr(strcmp(Tr.unit, P.unit{i}) & Tr.stim_number == P.stim_number(i), :);
    s = R.spikes(strcmp(R.spikes.unit, P.unit{i}) & R.spikes.stim_number == P.stim_number(i), :);
    bad(1) = bad(1) + (sum(t.n_window) ~= P.n_response(i));
    bad(2) = bad(2) + (abs(mean(t.n_window) - P.mean_post(i)) > 1e-9);
    bad(3) = bad(3) + (median(t.n_post) ~= P.median_post(i));
    bad(4) = bad(4) + (abs(signtest(t.n_base, t.n_post, 'Tail', 'left') - P.p_value_sign(i)) > 1e-9);
    bad(5) = bad(5) + (nnz(s.t_ms >= P.win_lo(i) & s.t_ms <= P.win_hi(i)) ~= P.n_response(i));
end
fprintf('rows %d | mismatches: n_response %d, mean_post %d, median_post %d, sign p %d, ticks-in-window %d\n', height(P), bad);
disp(U(:, {'unit','unit_kind','ref_unit','n_spikes','n_response_total','n_sign_p05','n_z_ge5','median_median_post'}));
