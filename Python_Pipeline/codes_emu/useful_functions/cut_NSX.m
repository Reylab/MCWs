function cut_NSX(folderpath, first_sample, last_sample, outputpath)
%cut_NSX cuts a segment of a given NSX folder
%   cut_NSX(folderpath, first_sample, last_sample, outputpath)
%   will take the parsed files in folderpath and create a new folder in 
%   outputpath with only the data between first_sample and last_sample 
%   (matlab indexing). It only edits the lts field in NSx.mat.

  nsxfiles=dir(['.' filesep '*.NC*']);


  load([folderpath filesep 'NSx.mat']);
  mkdir(outputpath);

  for i = 1:numel(nsxfiles)
    [~,name,ext] = fileparts(nsxfiles(i).name);
    selected = arrayfun(@(x) strcmp(name,x.output_name) && strcmp(ext,x.ext),NSx);
    NSx(selected).lts = last_sample - first_sample +1;
    fi = fopen(sprintf('%s%c%s',folderpath,filesep,nsxfiles(i).name),'r','l');
    x_raw = fread(fi,'int16=>int16');
    fclose(fi);
    fo = fopen(sprintf('%s%c%s',outputpath,filesep,nsxfiles(i).name),'w');
    x_raw = fwrite(fo,x_raw(first_sample:last_sample),'int16');
    fclose(fo);
  end
  save([outputpath filesep 'NSx.mat'], 'NSx','freq_priority','files')
end