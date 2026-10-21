S = '/mnt/data0/MCWs Sessions/matlab validation/single channel/';
a = load([S 'pt 11 emu 17 ch 328/NSX_TimeStamps.mat']); disp(a);
n = load([S 'pt 11 emu 17 ch 328/NSx.mat']); disp(n.NSx([n.NSx.chan_ID]==328));
d = dir([S 'pt 11 emu 17 ch 328/*.NC5']); fprintf('328 nc5 bytes %d -> int16 samples %d\n', d.bytes, d.bytes/2);
n2 = load([S 'pt 12 emu 18 ch 299/NSx.mat']); disp(n2.NSx([n2.NSx.chan_ID]==299));
d = dir([S 'pt 12 emu 18 ch 299/mRFUS02 raw_299.NC5']); fprintf('299 nc5 bytes %d -> int16 samples %d\n', d.bytes, d.bytes/2);
type('/home/ajb/Documents/wave_clus-master/Raw_data_readers/nc5_wc_reader.m')
