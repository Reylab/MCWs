function png_files = list_pngs_sorted(base_dir, patterns)
% LIST_PNGS_SORTED  Collects every file matching PATTERNS under BASE_DIR
% (each pattern resolved relative to base_dir, so 'gt/*.png' or a plain
% '*.png' both work), sorted alphabetically WITHIN each pattern -- used to
% build a page order for compile_pngs_to_pdf.m. Patterns are kept in the
% order given (so e.g. {'gt/*.png','rescue_sd1/*.png'} puts every GT page
% before every rescue_sd1 page, rather than interleaving them the way a
% single glob across both folders would).
%
% VARIABLES
%   base_dir    - input: folder the patterns are resolved against
%   patterns    - input: cellstr (or single char) of dir()-style patterns
%   png_files   - output: cellstr of full paths, in pattern order then
%                 alphabetical within each pattern
    if ischar(patterns), patterns = {patterns}; end
    png_files = {};
    for p = 1:numel(patterns)
        d = dir(fullfile(base_dir, patterns{p}));
        d = d(~[d.isdir]);
        names = sort({d.name});
        for i = 1:numel(names)
            idx = find(strcmp({d.name}, names{i}), 1);
            png_files{end+1,1} = fullfile(d(idx).folder, d(idx).name);
        end
    end
end
