function redo_runs(list_file, pattern)
% Redo validation runs from their own saved cfg, in a cleared folder.
addpath('/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline/validation_pipeline');
D = strtrim(strsplit(fileread(list_file), newline)); D = D(~cellfun(@isempty, D));
D = D(contains(D, pattern));
for i = 1:numel(D)
    s = load(fullfile(D{i}, 'validation_state.mat'), 'cfg'); cfg = s.cfg; cfg.stages = 'all';
    assert(strcmp(cfg.out_dir, D{i}), 'cfg out_dir %s ~= %s', cfg.out_dir, D{i});
    rmdir(D{i}, 's');
    run_validation(cfg);
    fprintf('[redo] %s: %s class %s | arm %s class %s\n', D{i}, cfg.gt.name, mat2str(cfg.gt.class), cfg.arms.name, mat2str(cfg.arms.class));
end
end
