function align_photo_ripple(ch_photo_ID)

%photoresistor
figure
ch_photo_ind = find(arrayfun(@(x) (x.chan_ID==ch_photo_ID),NSx));

sr = NSx(ch_photo_ind).sr;
lts = NSx(ch_photo_ind).lts;
conversion = NSx(ch_photo_ind).conversion;
ch_type = NSx(ch_photo_ind).ext;
output_name = NSx(ch_photo_ind).output_name;

f1 = fopen(sprintf('%s%s',output_name,ch_type),'r','l');
data_plot = fread(f1,lts,'int16=>double')*conversion;
fclose(f1);
    
plot((0:(lts-1))/sr,data_plot)
ylabel(NSx(ch_photo_ind).unit); xlabel('Time (s)');
title('photoresistor')
