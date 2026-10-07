function compile_pngs_to_pdf(png_files, pdf_path, page_captions)
% COMPILE_PNGS_TO_PDF  Stacks a list of already-rendered PNGs into ONE
% multi-page PDF, one PNG per page, in the given order -- so a whole
% thematic group of individually-saved PNGs (e.g. every stimulus-raster
% page, or every waveform-comparison figure) can be flipped through as a
% single document instead of opened one file at a time.
%
% VARIABLES
%   png_files     - input: cellstr of PNG paths, in the order they should
%                   appear as pages. Empty/missing files are skipped with a
%                   warning rather than erroring the whole PDF.
%   pdf_path      - input: output .pdf path (deleted first if it already
%                   exists -- exportgraphics' 'Append' mode would otherwise
%                   keep piling pages onto a stale PDF from a previous run)
%   page_captions - OPTIONAL: cellstr, same length as png_files, printed as
%                   a small caption strip above each page (e.g. the PNG's
%                   own filename, so you can still tell pages apart/find the
%                   source file). Omit or pass {} for no captions.
%   A/h/w         - loaded page image + its pixel size (sets the page's own
%                   aspect ratio so images aren't stretched/cropped)
%   fig/ax        - throwaway figure+axes used only to hand the image to
%                   exportgraphics, closed immediately after each page
    if nargin < 3, page_captions = {}; end
    if exist(pdf_path,'file'), delete(pdf_path); end
    if isempty(png_files)
        return;
    end
    out_dir = fileparts(pdf_path);
    if ~isempty(out_dir) && ~exist(out_dir,'dir'), mkdir(out_dir); end

    for i = 1:numel(png_files)
        if ~exist(png_files{i},'file')
            warning('compile_pngs_to_pdf:missing','skipping missing file: %s', png_files{i});
            continue;
        end
        A = imread(png_files{i});
        [h,w,~] = size(A);
        has_caption = ~isempty(page_captions);
        cap_frac = 0.03;

        fig = figure('Visible','off','Color','w','Units','pixels','Position',[100 100 w h]);
        if has_caption
            ax = axes(fig,'Position',[0 0 1 1-cap_frac]);
        else
            ax = axes(fig,'Position',[0 0 1 1]);
        end
        imshow(A,'Parent',ax,'Border','tight');
        if has_caption
            annotation(fig,'textbox',[0 1-cap_frac 1 cap_frac],'String',page_captions{i}, ...
                'Interpreter','none','FontSize',9,'EdgeColor','none', ...
                'HorizontalAlignment','center','VerticalAlignment','middle');
        end
        try
            exportgraphics(fig, pdf_path, 'Append', true, 'ContentType','image', 'Resolution',150);
        catch ME
            warning('compile_pngs_to_pdf:export_failed','failed on %s: %s', png_files{i}, ME.message);
        end
        close(fig);
    end
end
