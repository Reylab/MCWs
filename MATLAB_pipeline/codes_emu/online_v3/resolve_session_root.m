function root = resolve_session_root()
% Returns the session root directory — the folder containing NSx.mat.
% Works whether called from the root itself or from a spikes_* / times_* subfolder.

    if exist(fullfile(pwd, 'NSx.mat'), 'file')
        root = pwd;
        return;
    end
    
    one_up = fileparts(pwd);
    if exist(fullfile(one_up, 'NSx.mat'), 'file')
        root = one_up;
        return;
    end
    
    error(['resolve_session_root: Could not find NSx.mat in pwd (%s) ' ...
           'or one level up (%s). Please run from the session root ' ...
           'or a direct subfolder of it.'], pwd, one_up);
end