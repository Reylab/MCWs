addpath('/home/ajb/Documents/GitHub/codes_emu/wave_clus_reylab/batch_files');
addpath('/home/ajb/Documents/GitHub/codes_emu/useful_functions');
SC = '/mnt/data0/MCWs Sessions/matlab validation/single channel';
C = {'pt 11 emu 17 ch 328', 328, 'mLFUS08 raw_328.NC5'; 'pt 10 emu 7 ch 368', 368, 'mRFUS07 raw_368.NC5'};
for i = 1:size(C, 1)
    d = fullfile(SC, C{i,1});
    P = load(fullfile(d, 'pre_processing_info.mat'));
    fprintf('\n== %s: pre_processing_info fields: %s\n', C{i,1}, strjoin(fieldnames(P)', ', '));
    pi_ = P.process_info;
    ids = [pi_.chID];
    k = find(ids == C{i,2});
    fprintf('  %d entries, chID range %d..%d, entry for %d: %s\n', numel(pi_), min(ids), max(ids), C{i,2}, mat2str(k));
    if isempty(k), continue; end
    e = pi_(k(1));
    fprintf('  fields: %s\n  SOS %s, G %s, any NaN/Inf in SOS: %d, G = %s\n', strjoin(fieldnames(e)', ', '), mat2str(size(e.SOS)), mat2str(size(e.G)), ...
        any(~isfinite(e.SOS(:))), mat2str(e.G, 4));
    fid = fopen(fullfile(d, C{i,3}), 'r', 'l'); x = fread(fid, 30000*60, 'int16=>double')'/4; fclose(fid);
    y = fast_filtfilt(e.SOS, e.G, x);
    fprintf('  notch-filtered 60 s: std %.3g, NaN %d, max|y| %.3g\n', std(y, 'omitnan'), nnz(isnan(y)), max(abs(y)));
    [b, a] = ellip(4, 0.1, 40, [300 3000]*2/30000);
    [sos, g] = tf2sos(b, a); z = fast_filtfilt([e.SOS; sos], g * e.G, x);
    thr = 5 * median(abs(z)) / 0.6745;
    fprintf('  + 300-3000 Hz: noise SD %.3g uV, 5-SD threshold %.3g, crossings below -thr: %d\n', median(abs(z))/0.6745, thr, nnz(diff(z < -thr) == 1));
end
