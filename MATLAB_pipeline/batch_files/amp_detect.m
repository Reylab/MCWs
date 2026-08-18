function [spikes,thr,index,remove_counter] = amp_detect(x, par, varargin)

p = inputParser;
addRequired(p, 'x');
addRequired(p, 'par', @isstruct);
addParameter(p, 'use_ref', false, @(arg) islogical(arg) || isnumeric(arg));

% Handle legacy single string flag
if numel(varargin) == 1 && (ischar(varargin{1}) || isstring(varargin{1})) && strcmpi(varargin{1}, 'use_ref')
    varargin = {'use_ref', true};
end

parse(p, x, par, varargin{:});
x = double(p.Results.x);
par = p.Results.par;
use_ref = p.Results.use_ref;

if size(x,2) > size(x,1); x = x(:); end

sr     = par.sr;
w_pre  = par.w_pre;
w_post = par.w_post;

% evaluate refractory period
if use_ref
    if isfield(par,'ref_ms')
        ref = floor(par.ref_ms * par.sr / 1000);
    else
        ref = par.ref;
    end
else
    ref = 0;
end

if ~isfield(par,'detection')
    par.detection = 'neg';
end
detect = par.detection;

stdmin = par.stdmin;
stdmax = par.stdmax;

% look for custom search back window duration
if isfield(par,'search_back_ms')
    search_back = max(0, round(par.search_back_ms * sr / 1000));
else
    search_back = round(0.3333 * sr / 1000);
end
% look for custom search forward window duration
if isfield(par,'search_forward_ms')
    search_forward = max(1, round(par.search_forward_ms * sr / 1000));
else
    search_forward = round(0.3333 * sr / 1000);
end

N        = length(x);
pad_samp = min(round(1.0 * sr), floor(N/2));

x_pad = [x(pad_samp:-1:1); x; x(end:-1:end-pad_samp+1)];

% filter sorting matrix
if par.sort_order > 0
    xf_pad = filt_signal(x_pad, par.sort_order, par.sort_fmin, par.sort_fmax, sr, par);
else
    if par.preprocessing && ~isempty(par.process_info)
        xf_pad = fast_filtfilt(par.process_info.SOS, par.process_info.G, x_pad);
    else
        xf_pad = x_pad;
    end
end

% filter detection matrix
if par.detect_order > 0
    xfd_pad = filt_signal(x_pad, par.detect_order, par.detect_fmin, par.detect_fmax, sr, par);
else
    if par.preprocessing && ~isempty(par.process_info)
        xfd_pad = fast_filtfilt(par.process_info.SOS, par.process_info.G, x_pad);
    else
        xfd_pad = xf_pad;
    end
end

xf        = xf_pad(pad_samp + 1 : pad_samp+N);
xf_detect = xfd_pad(pad_samp + 1 : pad_samp+N);

noise_std_detect = median(abs(xf_detect)) / 0.6745;
noise_std_sorted = median(abs(xf))        / 0.6745;
thr    = stdmin * noise_std_detect;
thrmax = stdmax * noise_std_sorted;

pre_safe  = w_pre  + 1;
post_safe = w_post + 3;

index = [];
last_accepted = -inf;

switch detect
    case 'neg'
        % track negative threshold edge crossings
        above = xf_detect(1:end-1) > -thr;
        below = xf_detect(2:end)   <= -thr;
        crossings = find(above & below) + 1;

    case 'pos'
        % track positive threshold edge crossings
        below_p = xf_detect(1:end-1) < thr;
        above_p = xf_detect(2:end)   >= thr;
        crossings = find(below_p & above_p) + 1;

    case 'both'
        % track both polarity threshold edge crossings
        above_b  = xf_detect(1:end-1) > -thr;
        below_b  = xf_detect(2:end)   <= -thr;
        cross_neg = find(above_b & below_b) + 1;

        below_p2 = xf_detect(1:end-1) < thr;
        above_p2 = xf_detect(2:end)   >= thr;
        cross_pos = find(below_p2 & above_p2) + 1;

        crossings = sort([cross_neg; cross_pos]);
end

if isempty(crossings)
    spikes = zeros(0, w_pre+w_post);
    remove_counter = 0;
    return
end

for i = 1:length(crossings)
    tc = crossings(i);

    % bypass if within refractory limits
    if tc <= last_accepted + ref
        continue
    end

    s = max(1, tc - search_back);
    e = min(N, tc + search_forward);
    if e <= s; continue; end

% Stage 1: locate a candidate peak on the DETECTION signal within the
    % crossing's own search window (same window amp_detect has always used).
    tp = s + find_extremum_loc(xf_detect(s:e), detect) - 1;

    % Stage 2: re-center the search on the SORTING signal around the
    % stage-1 candidate, not the raw crossing. xf and xf_detect are two
    % differently-ordered filters on the same passband (sort_order vs
    % detect_order); their decay trajectories can disagree by a few
    % thousandths right at the threshold, letting xf_detect re-cross
    % spuriously on the tail of a spike already found. A window anchored
    % only to that raw crossing can land too far from the true peak to
    % ever reach it (see refract_viol investigation); re-centering on the
    % stage-1 candidate gives the search a second, wider-reaching chance,
    % so both crossings converge on the same true peak and get caught by
    % the last_accepted check below instead of producing two detections.
    s2 = max(1, tp - search_forward);
    e2 = min(N, tp + search_forward);
    refined = s2 + find_extremum_loc(xf(s2:e2), detect) - 1;

    if refined <= last_accepted + 2
        continue
    end
    % drop locations failing safety bounds
    if refined - pre_safe < 1 || refined + post_safe > N
        continue
    end

    index(end+1) = refined;   
    last_accepted = refined;
end

nspk = length(index);

ls     = w_pre + w_post;
spikes = zeros(nspk, ls + 4);
xf(N+1 : N+w_post+3) = 0;

remove_counter = 0;
for i = 1:nspk
    p = index(i);
    % invalidate large noise artifact events
    if max(abs(xf(p-w_pre : p+w_post))) < thrmax
        spikes(i,:) = xf(p-w_pre-1 : p+w_post+2);
    else
        remove_counter = remove_counter + 1;
    end
end

% clean out empty spikes rows
aux = find(spikes(:, w_pre) == 0);
spikes(aux,:) = [];
index(aux)    = [];

switch par.interpolation
    case 'n'
        % truncate outer padding rows
        spikes(:, end-1:end) = [];
        spikes(:, 1:2)       = [];
    case 'y'
        spikes = int_spikes(spikes, par);
end

end 


function loc = find_extremum_loc(win, detect)
    switch detect
        case 'pos'
            % extract index of maximum value
            [~, loc] = max(win);
        case 'neg'
            % extract index of minimum value
            [~, loc] = min(win);
        case 'both'
            % balance peak values prioritizing negative troughs
            [vmax, imax] = max(win);
            [vmin, imin] = min(win);
            if abs(vmin) >= vmax; loc = imin; else; loc = imax; end
    end
end

function filtered = filt_signal(x, order, fmin, fmax, sr, par)
    [b, a] = ellip(order, 0.1, 40, [fmin fmax]*2/sr);
    if par.preprocessing && ~isempty(par.process_info)
        [sos, g] = tf2sos(b, a);
        g   = g * par.process_info.G;
        sos = [par.process_info.SOS; sos];
        filtered = fast_filtfilt(sos, g, x);
    else
        filtered = fast_filtfilt(b, a, x);
    end
end