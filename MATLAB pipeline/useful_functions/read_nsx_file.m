    
function [x_raw, metadata] = read_nsx_file(fullfile)
    [folder,file,ext]=fileparts(fullfile);
    load([folder filesep 'NSx.mat'])

    selected = arrayfun(@(x) strcmp(file,x.output_name) && strcmp(ext,x.ext),NSx);

    if sum(selected)==0
        error('channel not found in NSx.mat')
    elseif length(nonzeros(selected))>1
        [posch,~] = max(selected);
    else
        posch = find(selected);
    end
        
    ch_type = NSx(posch).ext;
    lts = NSx(posch).lts;
    unit = NSx(posch).unit;
    label = NSx(posch).label;
    conversion = NSx(posch).conversion;
    output_name = NSx(posch).output_name;
    metadata  = NSx(posch);
    if isfield(NSx,'dc')
        dc = NSx(posch).dc;
    else
        dc=0;
    end
    if isempty(ch_type)
        warning('channel %d not parsed',channel)
        return
    end  
    
    f1 = fopen(sprintf('%s%c%s%s',folder,filesep,output_name,ch_type),'r','l');
    x_raw = fread(f1,'int16=>double')*conversion+dc;
    fclose(f1);

end