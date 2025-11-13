%add path

codes_emu_folder = fileparts(fileparts(mfilename('fullpath')));
%codes_emu_folder = 'F:\GitHub\codes_emu';

addpath(codes_emu_folder);

folders = dir(codes_emu_folder);

normal_folders = folders(arrayfun(@(x) x.isdir && ~strcmp(x.name(1),'.'),folders));

custompath = reylab_custompath({normal_folders.name});