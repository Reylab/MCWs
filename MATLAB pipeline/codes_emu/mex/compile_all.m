

cfiles = dir('*.c');
for i=1:length(cfiles)
    mex(cfiles(i).name)
end
cd('performance')
FilterM
cd('..')
cd('isosplit5')
compile_mex_isosplit5
cd('..')


