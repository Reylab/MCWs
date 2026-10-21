function run_wave_clus_gt_notch(chf)
% wave_clus (wave_clus-master, preprocessing on) in <channel>/wave_clus notch with GT's own notch:
% the run folder gets a pre_processing_info.mat holding GT's par.process_info (from GT's spikes file),
% so no channel-level preprocessing/ or pre_processing_info.mat is ever read.
P = mcws_paths(); SC = P.single_channel;   % folders: mcws_paths.m
C = fullfile(SC, chf);
t = regexp(chf, 'pt (\d+) emu (\d+) ch (\d+)', 'tokens', 'once'); pt = str2double(t{1}); emu = str2double(t{2}); ch = str2double(t{3});
nc = dir(fullfile(C, sprintf('* raw_%d.NC5', ch))); stem = nc(1).name(1:end - 4);
gtf = gt_spikes_file(C, pt, emu, stem, P.seeg_data);
G = load(gtf, 'par'); pinfo = G.par.process_info;
wc = fullfile(C, 'wave_clus notch'); assert(~exist(wc, 'dir'), '%s exists', wc); mkdir(wc);
links = {[stem '.NC5'], 'NSx.mat'};
if exist(fullfile(C, 'NSX_TimeStamps.mat'), 'file'), links{end + 1} = 'NSX_TimeStamps.mat'; end
for l = links, system(sprintf('ln -s "%s" "%s"', fullfile(C, l{1}), fullfile(wc, l{1}))); end
made = {};
if ~exist(fullfile(C, 'NSX_TimeStamps.mat'), 'file')
    N = load(fullfile(C, 'NSx.mat')); e = N.NSx([N.NSx.chan_ID] == ch);
    lts = e.lts; sr = e.sr; save(fullfile(wc, 'NSX_TimeStamps.mat'), 'lts', 'sr'); made{end + 1} = 'NSX_TimeStamps.mat';
end
process_info = pinfo; save(fullfile(wc, 'pre_processing_info.mat'), 'process_info');
restoredefaultpath; addpath(genpath(P.wave_clus));
cd(wc); Get_spikes([stem '.NC5']); Do_clustering([stem '_spikes.mat']);
for l = [links made {'pre_processing_info.mat'}], delete(fullfile(wc, l{1})); end
S = load(fullfile(wc, [stem '_spikes.mat']), 'par', 'index');
Gi = load(gtf); if isfield(Gi, 'index_all'), gi = Gi.index_all; else, gi = Gi.index; end
fid = fopen(fullfile(wc, 'gt_notch_check.txt'), 'w');
fprintf(fid, '%s\t%s\tspikes %d\tGT detections %d\tnotch = GT %d\tdetection identical to GT %d\n', chf, gtf, numel(S.index), numel(gi), ...
    isequal(S.par.process_info.SOS, pinfo.SOS) && isequal(S.par.process_info.G, pinfo.G), isequal(sort(S.index(:)), sort(gi(:))));
fclose(fid);
end

function f = gt_spikes_file(C, pt, emu, stem, seeg)
f = fullfile(C, 'hand selected', [stem '_spikes.mat']);
if exist(f, 'file'), P = load(f, 'par'); if isfield(P.par, 'process_info'), return; end, end
rec = dir(fullfile(seeg, sprintf('MCW-FH_%03d', pt), 'EMU', sprintf('EMU-%03d_*', emu))); rec = rec([rec.isdir]);
rec = rec(~contains({rec.name}, {'BAD'})); if numel(rec) > 1 && any(contains({rec.name}, '_sunil')), rec = rec(contains({rec.name}, '_sunil')); end
f = fullfile(rec(1).folder, rec(1).name, [stem '_spikes.mat']);
assert(exist(f, 'file') == 2, 'no GT spikes file for %s', C);
end
