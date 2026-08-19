%% GREEDY_MATCHED_PURSUIT_RESCUE.M
%
% Greedy matching-pursuit "spike rescue" for refractory-violation /
% overlapping-detection chains, plus noise characterization from raw
% continuous data.
%
% ------------------------------------------------------------------
% WHAT THIS DOES (per chain of ambiguous/overlapping detections)
% ------------------------------------------------------------------
%   1. Correlate every candidate unit template against the chain
%      waveform at every time-lag (a "matched filter" bank).
%   2. Take the single best (unit, lag) match.
%   3. If its score clears a noise-calibrated threshold, fit its
%      amplitude and SUBTRACT (not excise) that scaled template from
%      the chain waveform, leaving a continuous residual of the same
%      length.
%   4. Repeat steps 1-3 on the residual, with all templates eligible
%      again except where a same-unit refractory violation would
%      result.
%   5. Stop when the best remaining score is below threshold, or
%      MAX_SPIKES is reached.
%
% The threshold is not a magic number: it is calibrated by asking
% "how large could this same matched-filter score get from pure
% background noise alone?" via Monte-Carlo simulation using a noise
% model estimated from REAL quiet stretches of your raw signal
% (regions with no detected spike nearby), preferably taken from
% near the chain being rescued, since noise level/spectrum can drift
% over a recording.
%
% ------------------------------------------------------------------
% WHAT YOU NEED TO SUPPLY (this is deliberately NOT bundled in here)
% ------------------------------------------------------------------
%   - TEMPLATES: one waveform per unit, same convention (sample rate,
%     pre/post samples) as your raw signal. Build these however you
%     already do (e.g. trimmed mean of isolated spikes per cluster).
%   - RAW: a full-bandwidth continuous voltage trace at the recording's
%     real sample rate. NOTE: if you are working from a Wave_Clus
%     "*_spikes.mat" file, its `psegment` variable is a DECIMATED
%     preview trace (in the example file checked, ~1667 Hz vs. the
%     actual 30000 Hz recording) meant only for GUI plotting -- it is
%     NOT valid for noise/spectrum estimation at spike bandwidth
%     (300-3000 Hz). You need the real continuous recording.
%   - ALL_SPIKE_SAMPLES: sample indices (into RAW) of every detected
%     spike (sorted + unsorted / all threshold crossings you trust),
%     so quiet regions can be defined as "far from any of these".
%
% The demo at the bottom (DEMO_run_on_synthetic_data) fabricates all
% three so you can see the whole thing run end-to-end. Swap in your
% real data and skip the demo section in production use.
%
% Tested under GNU Octave 8.4 (no MATLAB-only toolboxes required:
% only base + statistics `quantile`, which Octave and base MATLAB
% both provide via the Statistics and Machine Learning Toolbox --
% see the note by QUANTILE_ below if you don't have it).
%
% ==================================================================

%% ------------------------------------------------------------------
%  DEMO (safe to delete once you wire in real data)
%  ------------------------------------------------------------------
if ~exist('SKIP_DEMO', 'var') || ~SKIP_DEMO
    DEMO_run_on_synthetic_data();
end


%% ==================================================================
%  TOP-LEVEL ORCHESTRATOR
%  ==================================================================
function result = rescue_chain(raw, sr, chain_range, templates, ...
        peak_offsets, all_spike_samples, cfg)
%RESCUE_CHAIN  Run greedy matched-pursuit rescue on one chain.
%
%   result = RESCUE_CHAIN(raw, sr, chain_range, templates, ...
%                          peak_offsets, all_spike_samples, cfg)
%
% Inputs
%   raw               : Nx1 full-bandwidth continuous raw trace
%   sr                : sample rate (Hz) of raw
%   chain_range       : [first_sample last_sample], 1-based, inclusive,
%                        indices into raw spanning the ambiguous chain
%                        (pad this a bit beyond the crossing times so
%                        template tails aren't truncated)
%   templates         : 1xK cell array of unit templates (each a column
%                        vector). All templates should already be in
%                        the same voltage units/scale as raw (i.e. NOT
%                        pre-normalized to unit norm, unless raw is
%                        too -- amplitude fitting assumes consistent
%                        scaling between the two).
%   peak_offsets      : 1xK vector, samples from templates{k}(1) to its
%                        reference/peak sample. Pass [] to auto-detect
%                        each template's own |max| sample.
%   all_spike_samples : vector of every known spike sample index (used
%                        to define "quiet" = far from any of these)
%   cfg               : struct, see DEFAULT_RESCUE_CFG(). Pass []
%                        to use all defaults, or DEFAULT_RESCUE_CFG()
%                        with a few fields overridden.
%
% Output: struct with fields
%   chain_raw          : the extracted chain waveform (n x 1)
%   spikes             : struct array, one row per inferred spike, with
%                         fields unit, tau, peak_sample (absolute index
%                         into raw), amplitude, score
%   residual           : n x 1 leftover waveform after all subtractions
%   reconstruction      : n x 1 sum of all subtracted (scaled) templates
%   threshold           : the calibrated matched-filter score threshold
%   accepted_scores     : scores of accepted spikes, in selection order
%   whitened_resid_var  : whitened residual variance per sample
%                         (~1 if the noise model and fit are both good;
%                         much >1 means something is off -- bad noise
%                         model, missing unit, wrong templates, etc.)
%   noise               : the noise model actually used
%   n_candidates        : number of (unit, lag) candidates in the bank

    if isempty(cfg)
        cfg = default_rescue_cfg();
    end

    chain_raw = raw(chain_range(1):chain_range(2));
    chain_raw = chain_raw(:);
    n = numel(chain_raw);

    if isempty(peak_offsets)
        peak_offsets = zeros(1, numel(templates));
        for k = 1:numel(templates)
            [~, ipk] = max(abs(templates{k}));
            peak_offsets(k) = ipk - 1;
        end
    end

    % --- Noise model, localized around this chain ---
    chain_center = round(mean(chain_range));
    radius_samples = round(cfg.noise_search_radius_sec * sr);
    noise = estimate_noise_model(raw, sr, all_spike_samples, ...
        struct( ...
            'exclude_ms',      cfg.exclude_ms, ...
            'n_acf',           cfg.n_acf, ...
            'ridge',           cfg.ridge, ...
            'search_center',   chain_center, ...
            'search_radius',   radius_samples, ...
            'min_run_samples', cfg.n_acf + 5));

    % --- Template bank + Monte-Carlo threshold for this chain length ---
    bank = build_template_bank(templates, peak_offsets, n, noise, cfg);
    threshold = calibrate_matched_filter_threshold(bank, cfg);

    refractory_samples = max(1, round(cfg.refractory_ms * 1e-3 * sr));

    % --- Greedy pursuit ---
    pursuit = greedy_matched_pursuit(chain_raw, bank, threshold, ...
        refractory_samples, cfg);

    result = pursuit;
    result.chain_raw    = chain_raw;
    result.threshold    = threshold;
    result.noise        = noise;
    result.n_candidates = size(bank.Aw, 2);
    % Report absolute sample positions, not chain-local ones.
    for i = 1:numel(result.spikes)
        result.spikes(i).peak_sample = result.spikes(i).peak_sample ...
            + chain_range(1) - 1;
    end
end


function cfg = default_rescue_cfg()
%DEFAULT_RESCUE_CFG  Tunable knobs, all in one place.
    cfg = struct();

    % --- Noise estimation ---
    cfg.exclude_ms  = 2.0;   % ms buffer excluded around every known spike
    cfg.n_acf       = 40;    % number of autocorrelation lags to estimate
    cfg.ridge       = 1e-4;  % diagonal loading fraction for stability
    cfg.noise_search_radius_sec = 5;  % search +/- this many sec around
                                       % the chain for quiet stretches
                                       % (use Inf to search the whole trace)

    % --- Matched-filter threshold calibration ---
    cfg.n_null = 5000;   % Monte-Carlo null draws
    cfg.alpha  = 1e-3;   % family-wise false-positive rate for the
                          % max-over-all-candidates null statistic

    % --- Greedy pursuit ---
    cfg.max_spikes    = 8;     % hard cap on inferred spikes per chain
    cfg.refractory_ms = 1.0;   % same-unit refractory period

    % --- Optional amplitude bounds, one row per unit: [lo hi]. ---
    % Leave empty for unconstrained (plain matching pursuit). If you
    % want e.g. "amplitude must be positive and below 3x template
    % norm", fill this in per unit before calling build_template_bank,
    % or just leave it empty -- unconstrained MP subtracts whatever
    % amplitude best explains the residual, which is the classic /
    % simplest version of what you described.
    cfg.amp_bounds = [];  % Kx2, or [] for unconstrained
end


%% ==================================================================
%  1. NOISE CHARACTERIZATION FROM RAW SIGNAL
%  ==================================================================
function noise = estimate_noise_model(raw, sr, spike_samples, cfg)
%ESTIMATE_NOISE_MODEL  Colored-noise autocorrelation from quiet stretches
%of a raw continuous trace.
%
%   noise = ESTIMATE_NOISE_MODEL(raw, sr, spike_samples, cfg)
%
% "Quiet" = at least cfg.exclude_ms away (in time) from every sample in
% spike_samples. Optionally restricted to a window around
% cfg.search_center (in samples) of +/- cfg.search_radius samples, so
% you can characterize noise LOCALLY around one chain rather than
% pooling the whole recording (useful if noise level/spectrum drifts).
%
% cfg fields (all required if calling directly; RESCUE_CHAIN fills
% these in for you):
%   exclude_ms, n_acf, ridge, search_center, search_radius,
%   min_run_samples
%
% Returns struct: acf (n_acf x 1), sigma, ridge, quiet_fraction,
% n_quiet_samples, n_runs, search_range

    raw = raw(:);
    N = numel(raw);

    if isempty(cfg.search_center) || isinf(cfg.search_radius)
        lo = 1; hi = N;
    else
        lo = max(1,  cfg.search_center - cfg.search_radius);
        hi = min(N,  cfg.search_center + cfg.search_radius);
    end

    excl_radius = max(1, round(cfg.exclude_ms / 1000 * sr));

    % Boolean "excluded" mask over [lo, hi], built by marking each
    % spike sample (that falls in range, expanded to the search window
    % by excl_radius on each side) as excluded.
    m = hi - lo + 1;
    excluded = false(m, 1);

    s = spike_samples(:);
    s = s(s >= lo - excl_radius & s <= hi + excl_radius);
    for i = 1:numel(s)
        a = max(lo, s(i) - excl_radius) - lo + 1;
        b = min(hi, s(i) + excl_radius) - lo + 1;
        excluded(a:b) = true;
    end

    quiet = ~excluded;

    % Find contiguous quiet runs long enough to use.
    d = diff([false; quiet; false]);
    run_starts = find(d == 1);
    run_ends   = find(d == -1) - 1;
    run_len    = run_ends - run_starts + 1;
    keep = run_len >= cfg.min_run_samples;
    run_starts = run_starts(keep);
    run_ends   = run_ends(keep);

    if isempty(run_starts)
        error(['estimate_noise_model:no_quiet_data', ...
            '\nNo quiet stretches >= %d samples found in range ' ...
            '[%d %d]. Widen noise_search_radius_sec, shorten n_acf, ' ...
            'or lower exclude_ms.'], cfg.min_run_samples, lo, hi);
    end

    runs = cell(numel(run_starts), 1);
    n_quiet_samples = 0;
    for r = 1:numel(run_starts)
        runs{r} = raw(lo - 1 + (run_starts(r):run_ends(r)));
        n_quiet_samples = n_quiet_samples + numel(runs{r});
    end

    pooled_mean = sum(cellfun(@sum, runs)) / n_quiet_samples;

    n_acf = cfg.n_acf;
    acf_sum   = zeros(n_acf, 1);
    acf_count = zeros(n_acf, 1);

    for r = 1:numel(runs)
        y = runs{r} - pooled_mean;
        L = numel(y);
        max_lag = min(n_acf, L) - 1;
        for lag = 0:max_lag
            acf_sum(lag + 1) = acf_sum(lag + 1) + sum(y(1:L-lag) .* y(1+lag:L));
            acf_count(lag + 1) = acf_count(lag + 1) + (L - lag);
        end
    end

    acf = acf_sum ./ max(acf_count, 1);
    acf(acf_count == 0) = 0;

    if ~isfinite(acf(1)) || acf(1) <= 0
        error('estimate_noise_model:bad_variance', ...
            'Estimated noise variance is non-positive/non-finite.');
    end

    noise = struct();
    noise.acf             = acf;
    noise.sigma            = sqrt(acf(1));
    noise.ridge            = cfg.ridge;
    noise.n_quiet_samples  = n_quiet_samples;
    noise.n_runs           = numel(runs);
    noise.quiet_fraction   = n_quiet_samples / m;
    noise.search_range     = [lo hi];
end


function C = build_noise_covariance(noise, n)
%BUILD_NOISE_COVARIANCE  Toeplitz colored-noise covariance of size n,
%from a NOISE struct's autocorrelation (Bartlett-tapered + ridge
%regularized for numerical stability), matching the ACF-based
%covariance construction used upstream in the Python noise model.
    acf = noise.acf;
    g = min(numel(acf), n);
    taper = 1 - (0:g-1)' / max(g, 1);
    taps = acf(1:g) .* taper;

    first_col = zeros(n, 1);
    first_col(1:g) = taps;

    C = toeplitz(first_col);
    C = C + noise.ridge * max(noise.sigma^2, 1e-12) * eye(n);
end


%% ==================================================================
%  2. TEMPLATE / LAG BANK  (every possible unit x time-shift)
%  ==================================================================
function bank = build_template_bank(templates, peak_offsets, n, noise, cfg)
%BUILD_TEMPLATE_BANK  Every (unit, lag) placement of every template
%within a window of length n, both in raw (A) and whitened (Aw) form.
%
% Precomputing Aw once here (rather than re-whitening every greedy
% iteration) is what keeps the pursuit loop cheap: whitening is linear,
% so only the running residual needs to be tracked/updated each
% iteration; the whitened template columns never change.

    K = numel(templates);
    if K == 0
        error('build_template_bank:no_templates', 'No templates given.');
    end

    C = build_noise_covariance(noise, n);
    Lc = chol(C, 'lower');

    A_cols  = {};
    unit_id = [];
    tau_vec = [];
    pos_vec = [];
    lo_vec  = [];
    hi_vec  = [];

    have_bounds = ~isempty(cfg.amp_bounds);

    for k = 1:K
        Tk = templates{k}(:);
        Lk = numel(Tk);
        if n < Lk
            error('build_template_bank:window_too_short', ...
                'Chain window (n=%d) shorter than template %d (L=%d).', ...
                n, k, Lk);
        end
        for tau = 0:(n - Lk)
            v = zeros(n, 1);
            v(tau+1:tau+Lk) = Tk;
            A_cols{end+1} = v; %#ok<AGROW>
            unit_id(end+1) = k; %#ok<AGROW>
            tau_vec(end+1) = tau; %#ok<AGROW>
            pos_vec(end+1) = tau + peak_offsets(k); %#ok<AGROW>
            if have_bounds
                lo_vec(end+1) = cfg.amp_bounds(k, 1); %#ok<AGROW>
                hi_vec(end+1) = cfg.amp_bounds(k, 2); %#ok<AGROW>
            end
        end
    end

    A = cat(2, A_cols{:});
    Aw = Lc \ A;   % triangular solve, applied once to the whole bank
    den = sum(Aw .^ 2, 1)';
    den = max(den, 1e-12);

    bank = struct();
    bank.A       = A;
    bank.Aw      = Aw;
    bank.units   = unit_id(:);
    bank.tau     = tau_vec(:);
    bank.pos     = pos_vec(:);
    bank.den     = den;
    bank.n       = n;
    bank.Lc      = Lc;
    if have_bounds
        bank.lo = lo_vec(:);
        bank.hi = hi_vec(:);
    else
        bank.lo = [];
        bank.hi = [];
    end
end


%% ==================================================================
%  3. NULL CALIBRATION  (how big can a pure-noise score get?)
%  ==================================================================
function [threshold, null_maxima] = calibrate_matched_filter_threshold(bank, cfg)
%CALIBRATE_MATCHED_FILTER_THRESHOLD  Monte-Carlo null distribution of
%the best-candidate matched-filter score under pure noise, using the
%SAME noise model and template bank as the real pursuit.
%
% Key trick: bank.Aw columns are already whitened (Aw = Lc \ A), and
% whitened Gaussian colored noise is just standard iid noise
% (x = Lc*z  =>  Lc \ x = z ~ N(0, I)). So we never need to synthesize
% colored noise at all -- draw z ~ N(0, I_n) directly and correlate it
% with the (already whitened) bank.

    n = bank.n;
    have_bounds = ~isempty(bank.lo);

    null_maxima = zeros(cfg.n_null, 1);
    batch = 500;
    done = 0;

    while done < cfg.n_null
        b = min(batch, cfg.n_null - done);
        Z = randn(n, b);

        num = bank.Aw' * Z;               % [n_candidates x b]
        amp = num ./ bank.den;            % broadcast over columns

        if have_bounds
            amp = min(max(amp, bank.lo), bank.hi);
        end

        gain = 2 * amp .* num - (amp .^ 2) .* bank.den;
        null_maxima(done+1:done+b) = max(gain, [], 1)';
        done = done + b;
    end

    threshold = quantile_(null_maxima, 1 - cfg.alpha);
end


function q = quantile_(x, p)
%QUANTILE_  Minimal quantile implementation (linear interpolation,
%same convention as MATLAB/Octave's default `quantile`), so this file
%has no Statistics-Toolbox dependency. If you DO have `quantile`
%available you can just call it directly instead.
    x = sort(x(:));
    n = numel(x);
    if n == 0
        q = NaN;
        return;
    end
    pos = p * n + 0.5;
    pos = min(max(pos, 1), n);
    lo = floor(pos);
    hi = ceil(pos);
    frac = pos - lo;
    q = x(lo) + frac * (x(hi) - x(lo));
end


%% ==================================================================
%  4. GREEDY MATCHING PURSUIT  (the core loop)
%  ==================================================================
function out = greedy_matched_pursuit(chain_raw, bank, threshold, ...
        refractory_samples, cfg)
%GREEDY_MATCHED_PURSUIT  Repeatedly pick the best-matching (unit, lag)
%candidate, fit its amplitude, and subtract it from the residual, in
%raw voltage space -- exactly the "correlate -> take best match ->
%subtract -> repeat" procedure, with all scoring done in a noise-
%whitened space so the "best match" and the stopping threshold both
%account for the real (colored) noise spectrum rather than raw
%correlation magnitude.

    chain_raw = chain_raw(:);
    n = numel(chain_raw);
    have_bounds = ~isempty(bank.lo);

    resid_w = bank.Lc \ chain_raw;   % whitened residual, updated in place
    reconstruction = zeros(n, 1);    % raw-space running sum of fits

    selected = [];
    spikes = struct('unit', {}, 'tau', {}, 'peak_sample', {}, ...
        'amplitude', {}, 'score', {});
    accepted_scores = [];

    for iter = 1:cfg.max_spikes
        num = bank.Aw' * resid_w;         % correlate residual w/ bank
        amp = num ./ bank.den;
        if have_bounds
            amp = min(max(amp, bank.lo), bank.hi);
        end
        scores = 2 * amp .* num - (amp .^ 2) .* bank.den;

        if ~isempty(selected)
            scores(selected) = -Inf;                  % no exact repeats
            for s = 1:numel(selected)
                j = selected(s);
                same_unit  = (bank.units == bank.units(j));
                too_close  = abs(bank.pos - bank.pos(j)) < refractory_samples;
                scores(same_unit & too_close) = -Inf;  % refractory
            end
        end

        [best_score, jbest] = max(scores);

        if ~isfinite(best_score) || best_score <= threshold
            break;   % candidates exhausted or nothing clears threshold
        end

        a = amp(jbest);

        resid_w = resid_w - a * bank.Aw(:, jbest);
        reconstruction = reconstruction + a * bank.A(:, jbest);

        selected(end+1) = jbest; %#ok<AGROW>
        accepted_scores(end+1) = best_score; %#ok<AGROW>

        spikes(end+1) = struct( ...
            'unit',        bank.units(jbest), ...
            'tau',         bank.tau(jbest), ...
            'peak_sample', bank.pos(jbest), ...
            'amplitude',   a, ...
            'score',       best_score); %#ok<AGROW>
    end

    % Sort by time for readability.
    if ~isempty(spikes)
        [~, order] = sort([spikes.peak_sample]);
        spikes = spikes(order);
    end

    residual = chain_raw - reconstruction;
    whitened_resid_var = (resid_w' * resid_w) / max(n, 1);

    out = struct();
    out.spikes              = spikes;
    out.selected            = selected;
    out.accepted_scores     = accepted_scores;
    out.reconstruction      = reconstruction;
    out.residual            = residual;
    out.whitened_resid_var  = whitened_resid_var;
    out.n_spikes            = numel(spikes);
end


%% ==================================================================
%  DEMO: synthetic raw trace + synthetic templates, end to end
%  ==================================================================
function DEMO_run_on_synthetic_data()
    fprintf('--- DEMO: synthetic colored noise + 2 overlapping units ---\n');
    rng(42);

    sr = 30000;
    T_sec = 20;
    N = round(T_sec * sr);

    % --- Fabricate colored background noise (AR(2) coloring of white
    %     noise, just so it is NOT flat/white -- a stand-in for your
    %     real raw trace) ---
    w = randn(N, 1) * 6;
    b = [1 0.6 -0.15];
    raw = filter(1, b, w);

    % --- Two toy unit templates (biphasic spike-like shapes) ---
    L = 64; wpre = 20;
    t = (0:L-1)';
    T1 = -80 * exp(-((t-wpre).^2)/(2*3^2)) + 25*exp(-((t-wpre-8).^2)/(2*5^2));
    T2 = -55 * exp(-((t-wpre).^2)/(2*4^2)) + 15*exp(-((t-wpre-10).^2)/(2*6^2));
    templates = {T1, T2};
    peak_offsets = [wpre-1, wpre-1];

    % --- Plant an overlapping pair (unit 1 then unit 2, 18 samples
    %     apart -- close enough that a threshold-crossing detector
    %     would likely only catch one of them) ---
    plant_center = round(N/2);
    tau1 = plant_center - (wpre-1);
    tau2 = tau1 + 18;
    raw(tau1+1:tau1+L) = raw(tau1+1:tau1+L) + T1;
    raw(tau2+1:tau2+L) = raw(tau2+1:tau2+L) + T2;

    % --- Pretend "all known spike samples" for noise exclusion: the
    %     two planted spikes plus a scattering of other unrelated
    %     detections elsewhere in the trace ---
    other_spikes = sort(randi([1 N], 40, 1));
    all_spike_samples = [plant_center; plant_center + 18; other_spikes];

    chain_range = [tau1 - 10, tau2 + L + 10];

    cfg = default_rescue_cfg();
    cfg.noise_search_radius_sec = 5;

    result = rescue_chain(raw, sr, chain_range, templates, ...
        peak_offsets, all_spike_samples, cfg);

    fprintf('Noise: sigma=%.2f, %d quiet samples in %d runs (%.1f%% of search window)\n', ...
        result.noise.sigma, result.noise.n_quiet_samples, ...
        result.noise.n_runs, 100*result.noise.quiet_fraction);
    fprintf('Bank: %d (unit,lag) candidates. Threshold = %.2f\n', ...
        result.n_candidates, result.threshold);
    fprintf('Ground truth: unit 1 @ sample %d, unit 2 @ sample %d\n', ...
        plant_center, plant_center + 18);
    fprintf('Inferred %d spike(s):\n', result.n_spikes);
    for i = 1:numel(result.spikes)
        sp = result.spikes(i);
        fprintf('  unit %d  sample %d  amplitude %.2f  score %.1f\n', ...
            sp.unit, sp.peak_sample, sp.amplitude, sp.score);
    end
    fprintf('Whitened residual variance: %.2f (want ~1)\n', ...
        result.whitened_resid_var);
end