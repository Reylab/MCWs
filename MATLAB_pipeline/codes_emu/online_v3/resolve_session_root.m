function root = resolve_session_root()
% Returns the session root directory — the folder containing NSx.mat.
% Works whether called from the root itself, a direct spikes_*/times_* subfolder,
% or a nested sibling folder like times_*_merged/.

    % Tier 1: Local Working Directory Check
    if exist(fullfile(pwd, 'NSx.mat'), 'file')
        root = pwd;
        return;
    end
    
    % Tier 2: One Level Up Check (Standard subfolders like spikes_* or times_*)
    one_up = fileparts(pwd);
    if exist(fullfile(one_up, 'NSx.mat'), 'file')
        root = one_up;
        return;
    end
    
    % Tier 3: Two Levels Up Check (Nested execution e.g., inside times_*_merged)
    two_up = fileparts(one_up);
    if ~isempty(two_up) && exist(fullfile(two_up, 'NSx.mat'), 'file')
        root = two_up;
        return;
    end
    
    % Fallback Error
    error(['resolve_session_root: Could not find NSx.mat in pwd (%s), ' ...
           'one level up (%s), or two levels up (%s). Please run from ' ...
           'the session root or a valid pipeline subfolder.'], pwd, one_up, two_up);
end