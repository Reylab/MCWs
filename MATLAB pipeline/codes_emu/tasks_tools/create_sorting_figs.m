function create_sorting_figs(labels,spikes,classes,filename,conversions, pre_fname,outfolder)
    if ~exist('conversions','var') || isempty(conversions)
        conversions = ones(numel(labels),1);
    end
    if ~exist('pre_fname','var') || isempty(pre_fname)
        pre_fname = 'IS';
    end
    if ~exist('outfolder','var') || isempty(outfolder)
        outfolder = '.';
    end
    
    disp('Creating sorting figures...')
    numfigs = numel(spikes);
    curr_fig = figure('Visible','Off');
    set(curr_fig, 'PaperUnits', 'inches', 'PaperType', 'A4', 'PaperPositionMode', 'auto','units','normalized','outerposition',[0 0 1 1],'RendererMode','manual','Renderer','painters')
    if isfield(get(curr_fig),'GraphicsSmoothing')
        set(curr_fig,'GraphicsSmoothing','off');
    end
    resolution = '-r150';

    ncolums = 4;
    nrows = 3;
    maxperfig = ncolums*nrows;
    colors = [[0.0 0.0 1.0];[1.0 0.0 0.0];[0.0 0.5 0.0];[0.620690 0.0 0.0];[0.413793 0.0 0.758621];[0.965517 0.517241 0.034483];
        [0.448276 0.379310 0.241379];[1.0 0.103448 0.724138];[0.545 0.545 0.545];[0.586207 0.827586 0.310345];
        [0.965517 0.620690 0.862069];[0.620690 0.758621 1.]];
    
    maxc = size(colors,1);
    
    for cnum = 1:numfigs
        f = ceil(cnum/(maxperfig));
        subpi = cnum - (f-1)*maxperfig;
        if f>1 && subpi==1
            print(curr_fig,'-dpng',sprintf('classes_%s_%s_fig%d.png',filename,pre_fname,f-1),resolution);
            clf(curr_fig)
        end
        ax = subplot(nrows,ncolums,subpi);
        hold on;
        grid on;
        ucl = unique(classes{cnum});
        ncl = numel(ucl);
        dot_plt = [];
        leg_labels = cell(ncl,1);
        ns = size(spikes{cnum},2);
        for i = 1: ncl
            if ucl(i)==0
                color = [0,0,0];
            else
                color = colors(mod(ucl(i)-1,maxc)+1,:);
            end
            idx  = classes{cnum}==ucl(i);
            
            msp = mean(spikes{cnum}(idx,:),1)*conversions(cnum);
            nidx = nnz(idx);
            leg_labels{i} = sprintf('cl:%d #%d',ucl(i),nidx);
            
            
            line(ax,1:ns,msp,'color',color,'LineWidth',1,'LineStyle', '-');
            dot_plt(end+1) = plot(ax,nan,nan,'.','color',color);
            if nidx>1
                ssp = std(spikes{cnum}(idx,:),1)*conversions(cnum);
                line(ax,1:ns,msp+ssp,'color',color,'LineWidth',0.5,'LineStyle', '--');
                line(ax,1:ns,msp-ssp,'color',color,'LineWidth',0.5,'LineStyle', '--');
            end
        end
        if ncl>0
            xlim(ax,[1, numel(msp)]);
        end
        legend(ax,dot_plt,leg_labels,'location','southeast','Box','off','interpreter','none');
        title(ax,labels{cnum},'interpreter','none')
        
    end
    print(curr_fig,'-dpng',fullfile(outfolder,sprintf('classes_%s_%s_fig%d.png',filename,pre_fname,f)),resolution);
    close(curr_fig)
    disp('Done')
end