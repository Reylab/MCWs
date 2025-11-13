function correlacion2(channels,fmin)
close all

% This function requires the parse_data_NSx to be applied before.
% Likewise, this code needs to be run inside a folder that contains a 
% folder called NK and RIP with their NSx respectively.

% Channels is a list of channel IDs as they were saved in the NSx.mat


% Define the path where the codes emu repository is located
[~,name] = system('hostname');
if contains(name,'BEH-REYLAB'), dir_base = '/home/user/share/codes_emu';
elseif contains(name,'TOWER-REYLAB') || contains(name,'RACK-REYLAB'),
    current_user = 'user';  % replace with appropriate user name
    dir_base = sprintf('/home/%s/Documents/GitHub/codes_emu',current_user);
elseif contains(name,'NSRG-HUB-15446'), dir_base = 'D:\codes_emu'; % Hernan's desktop
end

addpath(dir_base);
custompath = reylab_custompath({'wave_clus_reylab','NPMK','codes_for_analysis','mex','useful_functions','neuroshare' });


if ~exist('dirmain','var')
    dirmain = pwd;
end

% if ~exist('fmin','var')|| isempty(fmin),  fmin = [0:3]; end
if ~exist('fmin','var')|| isempty(fmin),  fmin = 0:3; end

currentFolder = pwd;
addpath('NK/',currentFolder);
addpath('RIP/',currentFolder);

NSx_NK = load('NK/NSx','NSx');
NSx_RIP = load('RIP/NSx','NSx');

lag_samples = [];
clabel = cell(length(channels));
flabel = cell(length(fmin));
info_file = fullfile(currentFolder,'aligned signals/', 'info.mat');

for j = 1:length(fmin)
    par_macro.sr = 2000;
    par_macro.detect_fmin = fmin(j);
    par_macro.detect_fmax = 500;
    par_macro.auto = 1;

    flabel{j}=fmin(j);

    for i = 1:length(channels)
        channel_number = string(extract(channels(i),lettersPattern|digitsPattern));
        channel = channel_number(1);
        number = channel_number(2);
        if startsWith(number,'0') && strlength(number)==2
            RIP_channel = channel+number;
            NK_channel = channel+extract(number,2);
        elseif ~startsWith(number,'0') && strlength(number)==2
            RIP_channel = channel+number;
            NK_channel = RIP_channel;
        elseif strlength(number)==1
            NK_channel = channel+number;
            RIP_channel =channel+'0'+number;
        end

        clabel{i} = RIP_channel;

        pos_chans_to_plot = find(arrayfun(@(x) endsWith(x.label,NK_channel), NSx_NK.NSx));

        if isempty(pos_chans_to_plot)
            warning('Channel %s not found',NK_channel)
            lag_samples(j,i) = NaN;
        else
            posch = pos_chans_to_plot(1);

            if NSx_NK.NSx(posch).is_micro
                par = par_micro;
                min_ref_per=1.5;
                ref = floor(min_ref_per*par.sr/1000);
                par.ref = ref;
            elseif NSx_NK.NSx(posch).sr ==2000
                par = par_macro;
            end

            min_record = 2;
            max_record1 = NSx_NK.NSx(posch).lts;


            if isfield(NSx_NK.NSx,'dc') && ~isempty(NSx_NK.NSx(posch).dc)
                dc = NSx_NK.NSx(posch).dc;
            else
                dc=0;
            end

            f1 = fopen(sprintf('%s%s',NSx_NK.NSx(posch).output_name,NSx_NK.NSx(posch).ext),'r','l');
            fseek(f1,(min_record-1)*2,'bof');
            Samples1 = fread(f1,(max_record1-min_record+1),'int16=>double')*NSx_NK.NSx(posch).conversion + dc;
            fclose(f1);

            pos_chans_to_plot = find(arrayfun(@(x) contains(x.label,RIP_channel), NSx_RIP.NSx));
            posch = pos_chans_to_plot(1);

            if NSx_RIP.NSx(posch).is_micro
                par = par_micro;
                min_ref_per=1.5;
                ref = floor(min_ref_per*par.sr/1000);
                par.ref = ref;
            elseif NSx_RIP.NSx(posch).sr ==2000
                par = par_macro;
            end

            min_record = 2;
            max_record2 = NSx_RIP.NSx(posch).lts;

            if isfield(NSx_RIP.NSx,'dc') && ~isempty(NSx_RIP.NSx(posch).dc)
                dc = NSx_RIP.NSx(posch).dc;
            else
                dc=0;
            end

            f1 = fopen(sprintf('%s%s',NSx_RIP.NSx(posch).output_name,NSx_RIP.NSx(posch).ext),'r','l');
            fseek(f1,(min_record-1)*2,'bof');
            Samples2 = fread(f1,(max_record2-min_record+1),'int16=>double')*NSx_RIP.NSx(posch).conversion + dc;
            fclose(f1);

            if fmin(j) == 0
                [C21,lag21] = xcorr(Samples1,Samples2);
                C21 = C21/max(C21);

                [~,I21] = max(C21);
                t21 = lag21(I21);

                if t21 < 0
                    Samples2 = Samples2(-t21:end);
                else
                    Samples2 = Samples2(t21:end);
                end

                if ~exist(fullfile(currentFolder,'aligned signals/'),'dir')
                    mkdir(currentFolder,'aligned signals/');
                end

                f = figure('visible','off');
                c(1) = subplot(2,1,1);
                plot(Samples1,'color','b')
                ylabel('NK')
                grid minor

                c(2) = subplot(2,1,2);
                plot(Samples2,'color','r')
                ylabel('Ripple')
                grid minor
                linkaxes(c,'x');

                title_text = sprintf('Aligned signals for channel:%s fmin:%d',RIP_channel,fmin(j));
                sgtitle(title_text,'interpreter','none')
                name = sprintf('aligned-signals_%s_fmin:%d.png',RIP_channel,fmin(j));
                saveas(f,fullfile(currentFolder,'aligned signals/', name),'png')

                lag_samples(j,i) = t21;
                clear [t21, Samples2, Samples1, rip_filt, posch, pos_chans_to_plot, par_micro, par_macro, par, nk_filt, min_record, max_record2, max_record1, M21, lag21, lag, I21, f1, dc, C21, c, b_orig, ans, a_orig]

            else
                [b_orig,a_orig]=ellip(4,0.1,40,[fmin(j) par.detect_fmax]*2/(par.sr));

                nk_filt = fast_filtfilt(b_orig,a_orig,Samples1);
                rip_filt = fast_filtfilt(b_orig,a_orig,Samples2);

                [C21,lag21] = xcorr(rip_filt,nk_filt);
                C21 = C21/max(C21);

                [~,I21] = max(C21);
                t21 = lag21(I21);

                rip_filt = rip_filt(t21:end);

                if ~exist(fullfile(currentFolder,'aligned signals/'),'dir')
                    mkdir(currentFolder,'aligned signals/');
                end

                f = figure('visible','off');
                c(1) = subplot(2,1,1);
                plot(nk_filt,'color','b')
                ylabel('NK')
                grid minor

                c(2) = subplot(2,1,2);
                plot(rip_filt,'color','r')
                ylabel('Ripple')
                grid minor
                linkaxes(c,'x');

                title_text = sprintf('Aligned signals for channel:%s fmin:%d',RIP_channel,fmin(j));
                sgtitle(title_text,'interpreter','none')
                name = sprintf('aligned-signals_%s_fmin:%d',RIP_channel,fmin(j));
                saveas(f,fullfile(currentFolder,'aligned signals/', name),'png')

                lag_samples(j,i) = t21;
                clear [t21, Samples2, Samples1, rip_filt, posch, pos_chans_to_plot, par_micro, par_macro, par, nk_filt, min_record, max_record2, max_record1, M21, lag21, lag, I21, f1, dc, C21, c, b_orig, ans, a_orig]
            end

        end

    end

    lag_sec = lag_samples*(1/2000);
    text = sprintf('fmin:%d\n Lag in sec:%d, lag in samples:%d',fmin(j),mean(lag_samples(j,:)),mean(lag_sec(j,:)));
    disp(text)

end

colormap = figure('visible','off');
imagesc(lag_samples)
colorbar

for i = 1:length(fmin)
    yline(i + 0.5, 'k'); 
    ytick = (1:i);
end

for j= 1:length(channels)
    xline(j + 0.5, 'k');
    xtick = (1:j);
end

xlabel('Channels');
ylabel('Fmin');
set(gca,'XTick',xtick,'XTickLabel',clabel,'TickLength',[0, 0])
set(gca,'YTick',ytick,'YTickLabel',flabel,'TickLength',[0, 0])
saveas(colormap,fullfile(currentFolder,'aligned signals/', 'Colormap'),'png')


bplot = figure('visible','off');
boxplot(lag_samples)
saveas(bplot,fullfile(currentFolder,'aligned signals/', 'Boxplot'),'png')

save(info_file, 'lag_sec','lag_samples','channels','fmin')

