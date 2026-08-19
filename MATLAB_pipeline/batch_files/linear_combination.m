%% GREEDY_TEMPLATE_RESCUE_V2.M
%
% Greedy template-matching "spike rescue" for a chain of refractory
% violations / overlapping detections -- built directly from the
% procedure described in conversation, nothing carried over from any
% earlier Python design:
%
%   1. Correlate every candidate unit template against the chain.
%   2. Take the single best match. If it's good enough, fit its
%      amplitude and SUBTRACT that scaled template from the chain
%      (in place -- the chain stays one continuous waveform, nothing
%      is excised).
%   3. Repeat on the new residual, with all templates eligible again.
%   4. Stop when the best remaining match is bad, or there's nothing
%      left worth trying.
%
% "Good enough" is decided by a threshold built from the RAW SIGNAL
% itself: quiet stretches (far from any known spike, preferably near
% the chain being rescued) are pulled out, their autocorrelation is
% measured directly, and that autocorrelation tells you analytically
% how large a correlation score pure background noise could produce
% against a given template -- i.e. "a correlation of noise with
% itself" -- with no simulated trials involved. The quiet stretches'
% power spectrum is also computed, just to let you look at it.
%
% ------------------------------------------------------------------
% WHAT YOU SUPPLY
% ------------------------------------------------------------------
%   - TEMPLATES : your unit waveforms, in raw voltage units (not
%     normalized), one per unit.
%   - RAW       : full-bandwidth continuous raw trace, same sample
%     rate as the templates. (If this is coming from a Wave_Clus
%     "*_spikes.mat" file: its `psegment` field is a decimated preview
%     for the GUI, not full-bandwidth data -- don't use it here.)
%   - ALL_SPIKE_SAMPLES : sample indices of every known/trusted
%     detection, so "quiet" can be defined as "away from all of
%     these".
%
% See DEMO_run_on_synthetic_data() at the bottom for a runnable,
% self-contained example.
%
% ==================================================================

if ~exist('SKIP_DEMO', 'var') || ~SKIP_DEMO
    DEMO_run_on_synthetic_data();
end


%% ==================================================================
%  TOP-LEVEL ORCHESTRATOR
%  ==================================================================
function result = rescue_chain(raw, sr, chain_range, templates, ...
        peak_offsets, all_spike_samples, cfg)
%RESCUE_CHAIN  Run greedy template-match rescue on one chain.
%
%   result = RESCUE_CHAIN(raw, sr, chain_range, templates, ...
%                          peak_offsets, all_spike_samples, cfg)
%
% Inputs
%   raw               : Nx1 full-bandwidth continuous raw trace
%   sr                : sample rate (Hz)
%   chain_range       : [first_sample last_sample], 1-based inclusive
%                        indices into raw spanning the chain (pad a
%                        bit past the crossings so template tails
%                        aren't cut off)
%   templates         : 1xK cell array of unit templates (column
%                        vectors, raw voltage units)
%   peak_offsets      : 1xK samples-from-start-of-template to its
%                        reference peak. [] to auto-detect via |max|.
%   all_spike_samples : every known spike sample index, for defining
%                        "quiet" regions
%   cfg               : struct, see DEFAULT_RESCUE_CFG(). [] for
%                        defaults.
%
% Output: struct with
%   chain_raw       : the extracted chain waveform
%   spikes          : struct array (unit, tau, peak_sample, amplitude,
%                      score, z) -- score is the raw correlation,
%                      z = score / noise-predicted std for that unit
%   residual        : leftover waveform after all subtractions
%   reconstruction  : sum of all subtracted (scaled) templates
%   z_threshold     : the n_sigma cutoff used
%   sigma_c         : 1xK, noise-predicted std of the correlation
%                      score for each template (this is the piece
%                      built from the noise autocorrelation)
%   noise           : the noise model actually used (includes .psd)

    if isempty(cfg)
        cfg = default_rescue_cfg();
    end

    chain_raw = raw(chain_range(1):chain_range(2));
    chain_raw = chain_raw(:);

    K = numel(templates);
    if isempty(peak_offsets)
        peak_offsets = zeros(1, K);
        for k = 1:K
            [~, ipk] = max(abs(templates{k}));
            peak_offsets(k) = ipk - 1;
        end
    end

    % --- Characterize background noise from quiet stretches of raw,
    %     centered on this chain. ---
    chain_center = round(mean(chain_range));
    radius_samples = round(cfg.noise_search_radius_sec * sr);
    max_lag = max(cellfun(@numel, templates));  % only need ACF out to
                                                  % the longest template
    noise = estimate_noise_from_raw(raw, sr, all_spike_samples, ...
        struct( ...
            'exclude_ms',      cfg.exclude_ms, ...
            'max_lag',         max_lag, ...
            'search_center',   chain_center, ...
            'search_radius',   radius_samples, ...
            'min_run_samples', max_lag + 5));

    % --- For each template, "correlate the noise's self-correlation
    %     with the template" to get the std-dev the matched-filter
    %     score would have under pure noise: sigma_c(k)^2 =
    %     T_k' * Sigma_Lk * T_k, where Sigma_Lk is the LkxLk Toeplitz
    %     covariance built from the measured noise autocorrelation. ---
    sigma_c = zeros(1, K);
    energy  = zeros(1, K);
    for k = 1:K
        Tk = templates{k}(:);
        Lk = numel(Tk);
        Sigma_Lk = small_noise_covariance(noise.acf, Lk);
        sigma_c(k) = sqrt(max(Tk' * Sigma_Lk * Tk, eps));
        energy(k)  = Tk' * Tk;
    end

    refractory_samples = max(1, round(cfg.refractory_ms * 1e-3 * sr));

    pursuit = greedy_template_match(chain_raw, templates, ...
        peak_offsets, sigma_c, energy, cfg.n_sigma, ...
        refractory_samples, cfg.max_iter);

    result = pursuit;
    result.chain_raw   = chain_raw;
    result.z_threshold = cfg.n_sigma;
    result.sigma_c      = sigma_c;
    result.noise        = noise;
    for i = 1:numel(result.spikes)
        result.spikes(i).peak_sample = result.spikes(i).peak_sample ...
            + chain_range(1) - 1;
    end
end


function cfg = default_rescue_cfg()
%DEFAULT_RESCUE_CFG  Tunable knobs.
    cfg = struct();

    % --- Noise characterization ---
    cfg.exclude_ms  = 2.0;   % ms buffer excluded around every known spike
    cfg.noise_search_radius_sec = 5;  % search +/- this many sec around
                                       % the chain for quiet stretches
                                       % (Inf = whole trace)

    % --- Match-quality threshold ---
    % A candidate is "good enough" if its correlation score is more
    % than n_sigma standard deviations above what pure background
    % noise would be expected to produce against that template (per
    % the noise autocorrelation). This is a plain z-score cutoff, not
    % a simulated false-positive rate -- raise it if you see spurious
    % matches, lower it if real overlaps are being missed.
    cfg.n_sigma = 4;

    % --- Greedy loop ---
    cfg.refractory_ms = 1.0;  % same-unit refractory period
    cfg.max_iter       = 10;  % safety cap on iterations (the loop
                               % normally stops itself once matches go
                               % bad or nothing valid is left to try)
end


%% ==================================================================
%  1. NOISE CHARACTERIZATION FROM RAW SIGNAL
%  ==================================================================
function noise = estimate_noise_from_raw(raw, sr, spike_samples, cfg)
%ESTIMATE_NOISE_FROM_RAW  Find quiet (spike-free) stretches of a raw
%continuous trace and characterize them: autocorrelation ("noise
%correlated with itself") and power spectrum.
%
% "Quiet" = at least cfg.exclude_ms from every sample in
% spike_samples. Restricted to +/- cfg.search_radius samples around
% cfg.search_center if given, so the noise model reflects the area
% around one particular chain rather than the whole recording.
%
% Returns struct:
%   acf              : (max_lag+1) x 1 autocorrelation, lag 0..max_lag
%   sigma             : sqrt(acf(1)), i.e. noise std
%   psd_freq, psd_power : coarse power-spectrum diagnostic (Hz, power)
%   n_quiet_samples, n_runs, quiet_fraction, search_range

    raw = raw(:);
    N = numel(raw);

    if isempty(cfg.search_center) || isinf(cfg.search_radius)
        lo = 1; hi = N;
    else
        lo = max(1, cfg.search_center - cfg.search_radius);
        hi = min(N, cfg.search_center + cfg.search_radius);
    end

    excl_radius = max(1, round(cfg.exclude_ms / 1000 * sr));

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

    d = diff([false; quiet; false]);
    run_starts = find(d == 1);
    run_ends   = find(d == -1) - 1;
    run_len    = run_ends - run_starts + 1;
    keep = run_len >= cfg.min_run_samples;
    run_starts = run_starts(keep);
    run_ends   = run_ends(keep);

    if isempty(run_starts)
        error('estimate_noise_from_raw:no_quiet_data', [ ...
            'No quiet stretches >= %d samples found in range [%d %d]. ' ...
            'Widen noise_search_radius_sec or lower exclude_ms.'], ...
            cfg.min_run_samples, lo, hi);
    end

    runs = cell(numel(run_starts), 1);
    n_quiet_samples = 0;
    for r = 1:numel(run_starts)
        runs{r} = raw(lo - 1 + (run_starts(r):run_ends(r)));
        n_quiet_samples = n_quiet_samples + numel(runs{r});
    end

    pooled_mean = sum(cellfun(@sum, runs)) / n_quiet_samples;

    max_lag = cfg.max_lag;
    acf_sum   = zeros(max_lag + 1, 1);
    acf_count = zeros(max_lag + 1, 1);

    for r = 1:numel(runs)
        y = runs{r} - pooled_mean;
        L = numel(y);
        this_max_lag = min(max_lag, L - 1);
        for lag = 0:this_max_lag
            acf_sum(lag + 1)   = acf_sum(lag + 1) + sum(y(1:L-lag) .* y(1+lag:L));
            acf_count(lag + 1) = acf_count(lag + 1) + (L - lag);
        end
    end

    acf = acf_sum ./ max(acf_count, 1);
    acf(acf_count == 0) = 0;

    if ~isfinite(acf(1)) || acf(1) <= 0
        error('estimate_noise_from_raw:bad_variance', ...
            'Estimated noise variance is non-positive/non-finite.');
    end

    [psd_freq, psd_power] = quiet_power_spectrum(runs, sr);

    noise = struct();
    noise.acf              = acf;
    noise.sigma             = sqrt(acf(1));
    noise.psd_freq          = psd_freq;
    noise.psd_power         = psd_power;
    noise.n_quiet_samples   = n_quiet_samples;
    noise.n_runs            = numel(runs);
    noise.quiet_fraction    = n_quiet_samples / m;
    noise.search_range      = [lo hi];
end


function Sigma = small_noise_covariance(acf, L)
%SMALL_NOISE_COVARIANCE  LxL Toeplitz covariance built from a measured
%autocorrelation, Bartlett-tapered so longer (noisier-to-estimate)
%lags are downweighted -- this is what turns the raw ACF into the
%matrix used to get T'*Sigma*T below.
    g = min(numel(acf), L);
    taper = 1 - (0:g-1)' / max(g, 1);
    taps = acf(1:g) .* taper;

    first_col = zeros(L, 1);
    first_col(1:g) = taps;

    Sigma = toeplitz(first_col);
end


function [f, Pxx] = quiet_power_spectrum(runs, sr)
%QUIET_POWER_SPECTRUM  Coarse Welch-style power spectrum of the quiet
%stretches, purely as a diagnostic you can plot (e.g. `plot(f, Pxx)`)
%to sanity-check the noise -- e.g. spot 60 Hz line noise or check the
%recording's analog filter band. Not used by the rescue algorithm
%itself. No toolbox dependency (Hann window computed by hand).

    total_len = sum(cellfun(@numel, runs));
    nfft = 1024;
    while nfft > total_len && nfft > 64
        nfft = nfft / 2;
    end
    nfft = round(nfft);

    win = 0.5 - 0.5 * cos(2*pi*(0:nfft-1)' / (nfft-1));
    win_power = sum(win .^ 2);

    accum = zeros(nfft, 1);
    n_segs = 0;
    step = floor(nfft / 2);  % 50% overlap

    for r = 1:numel(runs)
        y = runs{r};
        L = numel(y);
        starts = 1:step:(L - nfft + 1);
        for s = starts
            seg = y(s:s+nfft-1) .* win;
            X = fft(seg);
            accum = accum + (abs(X) .^ 2);
            n_segs = n_segs + 1;
        end
    end

    if n_segs == 0
        f = [];
        Pxx = [];
        return;
    end

    Pxx_full = accum / (n_segs * sr * win_power);
    half = floor(nfft/2) + 1;
    Pxx = Pxx_full(1:half);
    Pxx(2:end-1) = 2 * Pxx(2:end-1);  % one-sided
    f = (0:half-1)' * (sr / nfft);
end


%% ==================================================================
%  2. GREEDY TEMPLATE MATCH  (the core loop)
%  ==================================================================
function out = greedy_template_match(chain_raw, templates, ...
        peak_offsets, sigma_c, energy, n_sigma, refractory_samples, max_iter)
%GREEDY_TEMPLATE_MATCH  Correlate every template against the chain,
%take the best match, subtract it (scaled to its best-fit amplitude)
%if it clears a noise-derived z-score threshold, repeat.

    chain_raw = chain_raw(:);
    n = numel(chain_raw);
    K = numel(templates);

    residual = chain_raw;
    reconstruction = zeros(n, 1);

    spikes = struct('unit', {}, 'tau', {}, 'peak_sample', {}, ...
        'amplitude', {}, 'score', {}, 'z', {});
    accepted_positions = [];  % [unit, peak_sample] rows already accepted

    for iter = 1:max_iter
        best_z = -Inf;
        best_k = 0; best_tau = 0; best_c = 0;

        for k = 1:K
            Tk = templates{k}(:);
            Lk = numel(Tk);
            n_lags = n - Lk + 1;
            if n_lags < 1
                continue;
            end

            for tau = 0:(n_lags - 1)
                peak_pos = tau + peak_offsets(k);

                if ~isempty(accepted_positions)
                    same_unit = accepted_positions(:, 1) == k;
                    too_close = abs(accepted_positions(:, 2) - peak_pos) ...
                        < refractory_samples;
                    if any(same_unit & too_close)
                        continue;  % refractory-blocked, skip this candidate
                    end
                end

                window = residual(tau+1:tau+Lk);
                c = Tk' * window;
                z = c / sigma_c(k);

                if z > best_z
                    best_z = z;
                    best_k = k;
                    best_tau = tau;
                    best_c = c;
                end
            end
        end

        if ~isfinite(best_z) || best_z <= n_sigma
            break;  % matches are bad, or nothing valid left to try
        end

        Tk = templates{best_k}(:);
        Lk = numel(Tk);
        a = best_c / energy(best_k);

        residual(best_tau+1:best_tau+Lk) = residual(best_tau+1:best_tau+Lk) - a*Tk;
        reconstruction(best_tau+1:best_tau+Lk) = reconstruction(best_tau+1:best_tau+Lk) + a*Tk;

        peak_pos = best_tau + peak_offsets(best_k);
        accepted_positions(end+1, :) = [best_k, peak_pos]; %#ok<AGROW>

        spikes(end+1) = struct( ...
            'unit',        best_k, ...
            'tau',         best_tau, ...
            'peak_sample', peak_pos, ...
            'amplitude',   a, ...
            'score',       best_c, ...
            'z',           best_z); %#ok<AGROW>
    end

    if ~isempty(spikes)
        [~, order] = sort([spikes.peak_sample]);
        spikes = spikes(order);
    end

    out = struct();
    out.spikes          = spikes;
    out.reconstruction  = reconstruction;
    out.residual        = residual;
    out.n_spikes        = numel(spikes);
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

    % Colored background noise (AR(2) coloring of white noise) --
    % stand-in for a real raw trace.
    w = randn(N, 1) * 6;
    b = [1 0.6 -0.15];
    raw = filter(1, b, w);

    % Two toy unit templates.
    L = 64; wpre = 20;
    t = (0:L-1)';
    T1 = -80 * exp(-((t-wpre).^2)/(2*3^2)) + 25*exp(-((t-wpre-8).^2)/(2*5^2));
    T2 = -55 * exp(-((t-wpre).^2)/(2*4^2)) + 15*exp(-((t-wpre-10).^2)/(2*6^2));
    templates = {T1, T2};
    peak_offsets = [wpre-1, wpre-1];

    % Plant an overlapping pair: unit 1 then unit 2, 18 samples apart.
    plant_center = round(N/2);
    tau1 = plant_center - (wpre-1);
    tau2 = tau1 + 18;
    raw(tau1+1:tau1+L) = raw(tau1+1:tau1+L) + T1;
    raw(tau2+1:tau2+L) = raw(tau2+1:tau2+L) + T2;

    other_spikes = sort(randi([1 N], 40, 1));
    all_spike_samples = [plant_center; plant_center + 18; other_spikes];

    chain_range = [tau1 - 10, tau2 + L + 10];

    cfg = default_rescue_cfg();
    result = rescue_chain(raw, sr, chain_range, templates, ...
        peak_offsets, all_spike_samples, cfg);

    fprintf('Noise: sigma=%.2f, %d quiet samples in %d runs (%.1f%% of search window)\n', ...
        result.noise.sigma, result.noise.n_quiet_samples, ...
        result.noise.n_runs, 100*result.noise.quiet_fraction);
    fprintf('Per-template noise-predicted score std (sigma_c): %s\n', ...
        mat2str(result.sigma_c, 4));
    fprintf('z-threshold = %.1f\n', result.z_threshold);
    fprintf('Ground truth: unit 1 @ sample %d, unit 2 @ sample %d\n', ...
        plant_center, plant_center + 18);
    fprintf('Inferred %d spike(s):\n', result.n_spikes);
    for i = 1:numel(result.spikes)
        sp = result.spikes(i);
        fprintf('  unit %d  sample %d  amplitude %.2f  z=%.1f\n', ...
            sp.unit, sp.peak_sample, sp.amplitude, sp.z);
    end
    fprintf('(Power spectrum also available: result.noise.psd_freq / .psd_power)\n');
end