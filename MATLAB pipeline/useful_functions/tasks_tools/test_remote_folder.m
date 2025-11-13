function status=test_remote_folder(remote_folder)

if ispc
    status = boolean(exist(remote_folder,'dir'));
elseif isunix
    status = false;
    fid = fopen('/proc/mounts');
    tline = fgetl(fid);
    while ischar(tline)
        if contains(tline, remote_folder)
            status = true;
            break
        end
        tline = fgetl(fid);
    end
    fclose(fid);
else
    error('function test_acq_folder not tested for mac')  
end

end