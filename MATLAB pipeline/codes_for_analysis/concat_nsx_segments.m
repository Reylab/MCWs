function concat_nsx_segments(nsx_folder,segments,output_folder)


if ~exist(output_folder,'dir')
    mkdir(output_folder)
end

load([nsx_folder filesep 'NSx.mat'])

for fi = 1:length(NSx)

    fname = [NSx(fi).output_name NSx(fi).ext];

    f1 = fopen([nsx_folder filesep fname],'r','l');
    fo  = fopen([output_folder filesep fname],'w','l');
    lts = 0;
    
    for si=1:size(segments,1)
        start_sample = round(segments(si,1)*NSx(fi).sr);
        end_sample = round(segments(si,2)*NSx(fi).sr);
        nsamples = (end_sample-start_sample+1);
        
        fseek(f1,(start_sample-1)*2,'bof');
        Samples=fread(f1,nsamples,'int16=>int16');
        fwrite(fo,Samples,'int16');
        lts = lts + nsamples;
    end
    fclose(f1);
    fclose(fo);
    NSx(fi).lts = lts; 
end


save([output_folder filesep 'NSx.mat'], 'NSx','files','freq_priority', 'segments')


end