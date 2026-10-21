SC = '/mnt/data0/MCWs Sessions/matlab validation/single channel';
C = {'pt 11 emu 17 ch 328', 328, 'mLFUS08 raw_328.NC5'; 'pt 10 emu 7 ch 368', 368, 'mRFUS07 raw_368.NC5'};
for i = 1:size(C, 1)
    d = fullfile(SC, C{i,1});
    N = load(fullfile(d, 'NSx.mat')); NSx = N.NSx;
    e = NSx([NSx.chan_ID] == C{i,2});
    f = dir(fullfile(d, C{i,3}));
    fprintf('\n== %s\n  NSx entries: %d total, %d for chan %d\n', C{i,1}, numel(NSx), numel(e), C{i,2});
    if ~isempty(e)
        e = e(1); disp(e);
        fprintf('  file bytes %d, lts*2 = %d, equal: %d\n', f.bytes, e.lts*2, f.bytes == e.lts*2);
    end
    fid = fopen(fullfile(d, C{i,3}), 'r', 'l'); x = fread(fid, 30000*60, 'int16=>double')'/4; fclose(fid);
    fprintf('  first 60 s: mean %.2f, std %.2f, min %.1f, max %.1f, zeros %.1f%%\n', mean(x), std(x), min(x), max(x), 100*mean(x == 0));
    md = dir(fullfile(d, '*.mat')); fprintf('  .mat in folder: %s\n', strjoin({md.name}, ', '));
end
