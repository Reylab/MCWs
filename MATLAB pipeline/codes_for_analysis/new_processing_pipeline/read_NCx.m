function [Samples,metadata] = read_NCx(filename,start_sample,end_sample)
% filename needs to be the full path to the file

if ~exist('start_sample','var') || start_sample == 0,     start_sample = 1; end  %start acquiring at start_sample samples

load('NSx.mat')
[~,fname,~]=fileparts(filename);
selected = arrayfun(@(x) (strcmp(x.output_name,fname))*(find(freq_priority(end:-1:1)==x.sr)),NSx);
if sum(selected)==0
    error('channel not found in NSx.mat')
elseif length(nonzeros(selected))>1
    [posch,~] = max(selected);
else
    posch = find(selected);
end

if ~exist('end_sample','var'),     end_sample = NSx(posch).lts; end  %stop acquiring at end_sample samples
if isfield(NSx,'dc') && ~isempty(NSx(posch).dc)
    dc = NSx(posch).dc;
else
    dc=0;
end
metadata  = NSx(posch);
f1=fopen(filename,'r','l');
fseek(f1,(start_sample-1)*2,'bof');
Samples=fread(f1,(end_sample-start_sample+1),'int16=>double');
Samples = Samples*NSx(posch).conversion+dc; 
fclose(f1);
