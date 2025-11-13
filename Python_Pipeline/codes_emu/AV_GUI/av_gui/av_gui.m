function varargout = av_gui(varargin)
% AV_GUI MATLAB code for av_gui.fig
%      AV_GUI, by itself, creates a new AV_GUI or raises the existing
%      singleton*.
%
%      H = AV_GUI returns the handle to a new AV_GUI or the handle to
%      the existing singleton*.
%
%      AV_GUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in AV_GUI.M with the given input arguments.
%
%      AV_GUI('Property','Value',...) creates a new AV_GUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before av_gui_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to av_gui_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help av_gui

% Last Modified by GUIDE v2.5 04-Feb-2023 08:30:50

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @av_gui_OpeningFcn, ...
                   'gui_OutputFcn',  @av_gui_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before av_gui is made visible.
function av_gui_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to av_gui (see VARARGIN)

% Choose default command line output for av_gui
handles.output = hObject;
% Update handles structure
guidata(hObject, handles);

% UIWAIT makes av_gui wait for user response (see UIRESUME)
% uiwait(handles.av_gui);

function av_gui_CloseRequestFcn(hObject, eventdata, handles)
stop_av(handles);
delete(hObject);

% --- Outputs from this function are returned to the command line.
function varargout = av_gui_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure

varargout{1} = handles.output;
hFig = findobj('Tag', 'av_gui');
set(hFig, 'units','normalized','outerposition',[0 0 1 1]);
set(hFig, 'Visible','off');
disp('Loading...')
drawnow
start_av(hObject);
drawnow
disp('Done')
set(hFig, 'Visible','on');

% --- Executes on button press in pause_tb.
function pause_tb_Callback(hObject, eventdata, handles)
set(hObject,'Enable','off');
button_state = get(hObject,'Value');
     if button_state == 1
        if handles.par.mode == 'a'
            pause(handles.player)
        end
        stop(handles.timer_plot_loop);
        if handles.par.mode == 'v'
            set(handles.video_pause_tb,'Value',button_state);
            show_im_matrix(handles,0)
        end
     else
        if handles.par.mode == 'a'
            resume(handles.player)
        else
            set(handles.video_pause_tb,'Value',button_state);
            set(handles.view_fig,'units','pixels');
            pos = get(handles.view_fig,'position');
            set(handles.view_fig,'position',[pos(1)+floor(pos(3)/2) pos(2)+floor(pos(4)/2) handles.vidObj.Width handles.vidObj.Height]);            
        end
        set(handles.add_mark_pb,'Enable','off');
        if  strcmp(get(handles.timer_plot_loop,'Running'),'off');
            start(handles.timer_plot_loop);  
        end
        
        
     end
set(hObject,'Enable','on');

function show_im_matrix(handles,shift)
    shift = (getappdata(handles.view_fig,'shift_matrix')+shift)*abs(shift); % 0 => 0    1 => inc     -1 => dec
    setappdata(handles.view_fig,'shift_matrix',shift);
    video_frame = getappdata(handles.av_gui,'video_frame') + shift*(handles.par.fr_prev);
    hold(handles.video_axes, 'on')
    if shift == 0 %only when pause
        set(handles.view_fig,'units','normalize');
        set(handles.view_fig,'outerposition',[0 0 1 1]);
    end
    nfr = handles.par.fr_prev;
    w = handles.vidObj.Width ;
    h = handles.vidObj.Height;
    col = ceil(nfr/handles.par.nrows);
    for k = 1:nfr
        if (video_frame-nfr+k> 0)
            im = read(handles.vidObj,video_frame-nfr+k);
            image(w*mod(k-1,col),h*floor((k-1)/handles.par.nrows),im,'Parent',handles.video_axes)
        else
            im = read(handles.vidObj,video_frame)*0+256;
            image(w*mod(k-1,col),h*floor((k-1)/handles.par.nrows),im,'Parent',handles.video_axes)
        end
    end
    xlim(handles.video_axes,[0,w*(handles.par.nrows)]);
    ylim(handles.video_axes,[0 , h*(col)]);
    hold(handles.video_axes, 'off')
    drawnow
     
function stop_av(handles)
    stop(handles.timer_plot_loop);
    
    if handles.par.mode == 'a'
        stop(handles.player)
    else
        try
            close(handles.view_fig)
        end
    end
    
function plot_loop(obj, event,hObject)
    handles = guidata(hObject);
    sr = handles.sr;
    FRlength = handles.par.frame_len;
    if handles.par.mode =='a'
        new_chunck = (handles.player.CurrentSample + handles.audio_ref) >= sr*FRlength*handles.current_frame ;
    else
        video_frame = getappdata(hObject,'video_frame') + 1;
        setappdata(hObject,'video_frame',video_frame);
        new_chunck = video_frame > handles.vf_segment*handles.current_frame;

    end
    if  new_chunck
        setappdata(hObject,'keyPres',1);
        handles.selected_events = [];
        %hold(handles.multimedia_plot,'off');
        cla(handles.multimedia_plot);
        handles.current_frame = handles.current_frame + 1;
        %handles.current_frame = ceil((handles.player.CurrentSample + handles.audio_ref)/sr/FRlength);
        ind_beg = (floor(handles.current_frame-1)*FRlength*sr+1);
        ind_end = ceil(handles.current_frame*FRlength*sr);
        if ind_end > handles.lts
             ind_end = handles.lts;
        end
        
        if handles.par.mode =='a'
            plot(handles.multimedia_plot,handles.ejex_temp(ind_beg:3:ind_end),handles.audio(ind_beg:3:ind_end)); %isn't necesary too much precision
            samp_2_draw = handles.player.CurrentSample+ handles.audio_ref;
            curr_time = handles.ejex_temp(samp_2_draw);
        else
            handles.images(handles.vf_segment)= struct('cdata',[],'colormap',[]);
            for k = 1:handles.vf_segment
            	handles.images(k).cdata= read(handles.vidObj,handles.vf_segment*(handles.current_frame-1)+k);
            end
            image(handles.images(video_frame-handles.vf_segment*(handles.current_frame-1)).cdata,'Parent',handles.video_axes)
            curr_time = (video_frame-1)/handles.v_fr;
            
        end
        
        hold(handles.multimedia_plot,'on')
        borders = xlim(handles.multimedia_plot);
        gap = borders(2)-borders(1);
        if gap >FRlength
            gap = FRlength;
        end
        xlim(handles.multimedia_plot,[handles.ejex_temp(ind_beg) handles.ejex_temp(ind_beg)+gap]);
        ylim(handles.multimedia_plot,[-1 1])
        
        xlabel(handles.multimedia_plot,'Time [s]');
        if ~isempty(handles.events)
            if handles.par.mode == 'a'
                ev_in = (handles.events(:,2)<=handles.ejex_temp(ind_end)*1e6) & (handles.events(:,2)>=handles.ejex_temp(ind_beg)*1e6);
                ev_in = find(ev_in);
                for i = 1:length(ev_in)
                     e = ev_in(i);
                     nevent = handles.events(e,1);
                     plot(handles.multimedia_plot,[1  1]*handles.events(e,2)/1e6,[-1 1],'-.','linewidth',2,'color',handles.colors(mod(nevent,size(handles.colors,1)),:));
                     text(handles.events(e,2)/1e6,1,[num2str(nevent) ' '],'Parent',handles.multimedia_plot,'Color',handles.colors(mod(nevent,size(handles.colors,1)),:),'FontSize',12,'HorizontalAlignment','right','VerticalAlignment','top');
                end              
                
                
            else
                in_f = handles.vf_segment*(handles.current_frame-1)+1;
                end_f = handles.vf_segment*(handles.current_frame);
                ev_in = (handles.events(:,2)<=end_f) & (handles.events(:,2)>=in_f);
                ev_in = find(ev_in);
                for i = 1:length(ev_in)
                     e = ev_in(i);
                     nevent = handles.events(e,1);
                     plot(handles.multimedia_plot,[1  1]*handles.events(e,2)/handles.v_fr,[-1 1],'-.','linewidth',2,'color',handles.colors(mod(nevent,size(handles.colors,1)),:));
                     text(handles.events(e,2)/handles.v_fr,1,[num2str(nevent) ' '],'Parent',handles.multimedia_plot,'Color',handles.colors(mod(nevent,size(handles.colors,1)),:),'FontSize',12,'HorizontalAlignment','right','VerticalAlignment','top');
                end 
            end
            

        end
        handles.lineav = plot(handles.multimedia_plot,[curr_time curr_time],[-1 1],'linewidth',1,'color','k');
        if handles.par.show_fr || handles.par.show_raster || handles.par.show_lfp
            hold(handles.fr_axes,'off')
            handles.linefr = plot(handles.fr_axes,[curr_time curr_time],[0 handles.plot_counter],'linewidth',1,'color','k');
            hold(handles.fr_axes,'on')
            if length(handles.events) > 0
                for i = 1:length(ev_in)
                     e = ev_in(i);
                     nevent = handles.events(e,1);
                     plot(handles.fr_axes,[1  1]*handles.events(e,2)/1e6,[0 handles.plot_counter],':','linewidth',2,'color',handles.colors(mod(nevent,size(handles.colors,1)),:));
                     text(handles.events(e,2)/1e6,handles.plot_counter,[num2str(nevent) ' '],'Parent',handles.fr_axes,'Color',handles.colors(mod(nevent,size(handles.colors,1)),:),'FontSize',12,'HorizontalAlignment','right','VerticalAlignment','top');
                end
            end
            
            if handles.plot_counter>1
                yborders = 1:handles.plot_counter-1;
                plot(handles.fr_axes,[ones(1, length(yborders))*handles.ejex_temp(ind_beg); ones(1, length(yborders))*handles.ejex_temp(ind_end)],[yborders;yborders],'--','color',[0,0,0]+0.6);
            end
            xlim(handles.fr_axes,[handles.ejex_temp(ind_beg) handles.ejex_temp(ind_beg)+gap]);
            %xlim(handles.fr_axes,[handles.ejex_temp(ind_beg) handles.ejex_temp(ind_end)]);
            set(handles.fr_axes,'YTick',[])
            set(handles.fr_axes,'XTick',[])
            set(handles.fr_axes,'YMinorGrid','on')
            plot_counter = 1;
            for ch_n = 1: length(handles.par.channels)
                classes = handles.par.classes{ch_n};
                if strcmp(classes,'mu')
                    if handles.par.show_fr
                        plot(handles.fr_axes,handles.ejex_temp(ind_beg:2:ind_end),handles.ch_data{ch_n}.fr(ceil(ind_beg/2):ceil(ind_end/2)),'color',handles.colors(mod(plot_counter,size(handles.colors,1)),:),'linewidth',1);
                    end         
                    
                    if handles.par.show_raster
                        sp = handles.ejex_temp(find(handles.ch_data{ch_n}.sp_index(ind_beg:ind_end))+ind_beg);
                        plot(handles.fr_axes,[sp; sp], [ones(1, length(sp))*(plot_counter-1); ones(1, length(sp))*(plot_counter-1+0.2)],'color','k','linewidth',2);
                    end
                    plot_counter = plot_counter +1;
                else
                    if iscell(classes)
                        classes = cell2mat(classes); %fix when indexing handles.par.classes output cell
                    end
                    for cl_n = 1:length(classes)
                        if handles.par.show_fr
                            plot(handles.fr_axes,handles.ejex_temp(ind_beg:2:ind_end),handles.ch_data{ch_n}.fr(cl_n,ceil(ind_beg/2):ceil(ind_end/2)),'color',handles.colors(mod(plot_counter,size(handles.colors,1)),:),'linewidth',1);
                        end                    
                        if handles.par.show_raster
                            sp = handles.ejex_temp(find(handles.ch_data{ch_n}.index(cl_n,ind_beg:ind_end))+ind_beg);
                            plot(handles.fr_axes,[sp; sp], [ones(1, length(sp))*(plot_counter-1); ones(1, length(sp))*(plot_counter-1+0.2)],'color','k','linewidth',2);
                        end
                        plot_counter = plot_counter +1;
                    end
                end
            end
            
            if handles.plot_counter ~= 0
                ylim(handles.fr_axes,[0 handles.plot_counter])
            end
            if handles.par.show_lfp
                error('LFP not implemented')
            end

        end
        guidata(hObject,handles);
        setappdata(hObject,'keyPres',0);
    else
        if handles.par.mode =='a'
            newcurrent_time = handles.ejex_temp(handles.player.CurrentSample+handles.audio_ref);
        else
            newcurrent_time = (video_frame-1)/handles.v_fr+handles.par.tbeg;
            image(handles.images(video_frame-handles.vf_segment*(handles.current_frame-1)).cdata,'Parent',handles.video_axes)
        end
        set(handles.lineav, 'Xdata', [newcurrent_time newcurrent_time])
        borders = xlim(handles.multimedia_plot);
        gap = borders(2)-borders(1);
        if newcurrent_time > borders(2)
            xlim(handles.multimedia_plot,[newcurrent_time newcurrent_time+gap]);
        end
        if handles.par.show_fr || handles.par.show_raster || handles.par.show_lfp
            if newcurrent_time > borders(2)
                xlim(handles.fr_axes,[newcurrent_time newcurrent_time+gap]);
            end
            set(handles.linefr, 'Xdata', [newcurrent_time newcurrent_time])
        end
    end
    
    if handles.par.mode =='a'
        if handles.player.CurrentSample ~= 1
            set(handles.curr_time_label,'String',[num2str(handles.par.tbeg+(handles.player.CurrentSample + handles.audio_ref - 1)/handles.sr,'%3.3f') 's']);
        elseif handles.current_frame == handles.frame_max
            set(handles.curr_time_label,'String','End');
            stop(handles.player)
        end
    else
        if video_frame ~= handles.max_vf
            set(handles.curr_time_label,'String',num2str(video_frame));
        else
            set(handles.curr_time_label,'String','End');
        end
    end
    if handles.par.show_mav
        
        if handles.par.mode =='a'
            current_sample = ceil(handles.player.CurrentSample+handles.audio_ref);
        else
            current_sample = ceil((video_frame-1)/handles.v_fr*sr);
        end
        
        cc = 1;
        for ch_n = 1: length(handles.par.channels)
            ch = handles.par.channels(ch_n);
            classes = handles.par.classes{ch_n};
            if strcmp(classes,'mu')
                aux = handles.ch_data{ch_n}.mav(current_sample);
                handles.clabels{cc,2}.String = sprintf('%s w:%0.1f',handles.clabels{cc,1},aux);                cc = cc +1;
            else
                if iscell(classes)
                    classes = cell2mat(classes); %fix when indexing handles.par.classes output cell
                end
                for cl_n = 1:length(classes)
                    aux = handles.ch_data{ch_n}.mav(cl_n,current_sample);
                    handles.clabels{cc,2}.String = sprintf('%s w:%0.1f',handles.clabels{cc,1},aux);
                    cc = cc +1;
                end
            end
        end
    end
drawnow    

    
    
        
function start_av(hObject)
    clear functions % reset functions, force to reload set_parameters next
    handles = guidata(hObject);
    set(handles.pause_tb,'Value',1)
    handles.par = par_av_gui();
    handles.current_frame = 0;
    max_channels = length(handles.par.channels);
    plot_counter = 0;
    handles.selected_events = [];
    setappdata(hObject,'keyPres',0);
    xlim(handles.multimedia_plot,[0 handles.par.frame_len]);
    xlim(handles.multimedia_plot,'manual')
    ylim(handles.multimedia_plot,[-1 1])
    
    %check the parameters
        
    if isempty(handles.par.folder_base)
        handles.par.folder_base = pwd;
    end
    if isempty(handles.par.folder_resus) 
        handles.par.folder_resus = pwd;
    end
    inds_sep = strfind(handles.par.folder_base,filesep);
    handles.session = handles.par.folder_base(inds_sep(end)+1:end);
    load([fileparts(mfilename('fullpath')) filesep 'avgui_colors.mat'],'colors');    
    handles.colors = colors(2:end,:);
    handles.events = [];
    set(handles.add_mark_pb,'Enable','off');
    try
        ftxt = fopen(fullfile(handles.par.folder_base,'ImageNames.txt'));
        if ftxt == -1
            load(fullfile(handles.par.folder_base,'experiment.mat'))
            conceps =  experiment.ImageNames;
            exp_imnames = true;
        else
            conceps = textscan(ftxt,'%s','delimiter',sprintf('\n'));
            fclose(ftxt);
            exp_imnames = false;  
        end
              
    catch
        warning('ImageNames.txt or experiment.mat: Files Not Found');
        exp_imnames = false; 
        conceps = {''};
    end
    
    if length(conceps{1})== 0
        listboxItems{1} = '(1)';
    end
    for k = 1 : length(conceps{1}) %aca hay que leer bien, pedir ejemplo
        listboxItems{k} = ['(' num2str(k) ') ' conceps{1}{k} ];
    end
    
    set(handles.stim_lb, 'String', listboxItems);
    
    if handles.par.show_fr || handles.par.show_raster || (handles.par.mode == 'a') || handles.par.show_lfp
        ch_outnames = cell(max_channels,1);
        try
            if exist(fullfile(handles.par.folder_base,'NSx'),'file')
                lts = [];
                sr = [];
                if handles.par.mode == 'a'
                    if isnumeric(handles.par.audio_channel)
                        load(fullfile(handles.par.folder_base,'NSx'),'NSx','freq_priority');
                        selected = arrayfun(@(x) (x.chan_ID==handles.par.audio_channel)*(find(freq_priority(end:-1:1)==x.sr)),NSx);
                        if sum(selected)==0
                            error('channel not found in NSx.mat')
                        elseif length(nonzeros(selected))>1
                            [posch,~] = max(selected);
                        else
                            posch = find(selected);
                        end
                        sr(end+1) = NSx(posch).sr;
                        lts(end+1) = NSx(posch).lts;
                        audio_channel_file = [NSx(posch).output_name NSx(posch).ext];
                    else
                        sr(end+1) = handles.par.audio_sr;
                        aux_info = whos('-file',handles.par.audio_file,handles.par.audio_variable);
                        lts(end+1) = max(aux_info.size);
                    end
                end

                for ch_n = 1: max_channels
                    if exist('freq_priority','var') == false
                        load(fullfile(handles.par.folder_base,'NSx'),'NSx','freq_priority');
                    end                
                    ch = handles.par.channels(ch_n);
                    selected = arrayfun(@(x) (x.chan_ID==ch)*(find(freq_priority(end:-1:1)==x.sr)),NSx);
                    if sum(selected)==0
                        error('channel not found in NSx.mat')
                    elseif length(nonzeros(selected))>1
                        [posch,~] = max(selected);
                    else
                        posch = find(selected);
                    end
                    sr(end+1) = NSx(posch).sr;
                    lts(end+1) = NSx(posch).lts;
                    ch_outnames{ch_n}= NSx(posch).output_name;
                end

                if (max_channels>0) || (handles.par.mode == 'a')
                    lts = min(lts);
                    sr = unique(sr);
                    if length(sr)>1  
                        error('Multiple sampling rates')
                    end
                end
            else
                if handles.par.mode == 'a'
                    audio_channel_file = sprintf('NSX%d.NC5', handles.par.audio_channel);
                end
                load(fullfile(handles.par.folder_base,'NSX_TimeStamps.mat'),'sr','lts');
                for ch_n = 1: max_channels
                    ch_outnames{ch_n}=['NSX' num2str(handles.par.channels(ch_n))];
                end
            end
        catch ME
            rethrow(ME)
            load(fullfile(handles.par.folder_base,'NSX_TimeStamps'),'lts','sr');
            for ch_n = 1: max_channels
                ch_outnames{ch_n}=['NSX' num2str(handles.par.channels(ch_n))];
            end
        end
    else
        sr = 1;
        lts = 50000;
    end
    handles.sr= sr;
    if (sr * handles.par.tbeg) >= lts
        error('tbeg out of recording')
    end    
    if strcmp(handles.par.rec_length,'end') % rec_length in seconds. to decide from which point I playback the sound and show the FR
        tend = lts/sr;
        handles.par.rec_length = tend - handles.par.tbeg;
    else
        tend = handles.par.rec_length + handles.par.tbeg;
    end
    lts = floor(handles.par.rec_length*sr);

    handles.lts = lts;
    handles.frame_max = ceil(handles.par.rec_length/handles.par.frame_len);
    handles.ejex_temp = linspace(handles.par.tbeg,tend,lts);
    
    if handles.par.tbeg==0
        handles.min_record=1;
    else
        handles.min_record = ceil(sr * handles.par.tbeg);
    end
    
    if handles.par.show_fr || handles.par.show_raster || handles.par.show_mav
        ylim(handles.fr_axes,'manual');
        xlim(handles.fr_axes,'manual');
        handles.ch_data = cell(length(handles.par.channels),1);
        if handles.par.show_fr
            half_width_gauss = handles.par.alpha_gauss * handles.par.sigma_gauss;
            sample_period = 1000/sr; % sample period for the spike list - window convolution in ms/sample
            N_gauss = 2*round(half_width_gauss/sample_period)+1; % Number of points of the gaussian window
            int_window = gausswin(N_gauss, handles.par.alpha_gauss);
            int_window = 1000*int_window/sum(int_window)/sample_period;
        end
        handles.ch_data={};
        for ch_n = 1: max_channels
            ch = handles.par.channels(ch_n);
            load(fullfile(handles.par.folder_base,sprintf('times_%s.mat',ch_outnames{ch_n})),'cluster_class');
            if strcmp(handles.par.classes{ch_n},'all')
                handles.par.classes{ch_n} = 1 : max(cluster_class(:,1));
            end
            classes = handles.par.classes{ch_n};
            if strcmp(classes,'mu')
                 sorted_times = cluster_class(:,2);
                 sorted_times = sorted_times(sorted_times>(handles.par.tbeg*1000))-handles.par.tbeg*1000;
                 handles.ch_data{ch_n}.sp_index = zeros(lts,1);
                 sorted_samples = ceil(sorted_times*sr/1e3);
                 sorted_samples = sorted_samples(sorted_samples<=lts);
                 handles.ch_data{ch_n}.sp_index(sorted_samples) =1;
                 if handles.par.show_fr
                     n_spike_timeline = length(handles.ch_data{ch_n}.sp_index);
                     integ_timeline_stim = conv(handles.ch_data{ch_n}.sp_index, int_window);
                     handles.ch_data{ch_n}.fr = integ_timeline_stim(round(half_width_gauss/sample_period)+1:2:n_spike_timeline+round(half_width_gauss/sample_period));
                     handles.ch_data{ch_n}.frmax = prctile(handles.ch_data{ch_n}.fr,99);
                     handles.ch_data{ch_n}.fr = single(handles.ch_data{ch_n}.fr/handles.ch_data{ch_n}.frmax+(plot_counter)); % rescale 0 to 1, and add offset for plotting
                 end
                 if handles.par.show_mav
                    mav = conv(handles.ch_data{ch_n}.sp_index, ones(1,ceil(handles.par.window_len_mav*handles.sr)))/handles.par.window_len_mav;
                    handles.ch_data{ch_n}.mav = single(mav); % rescale 0 to 1, and add offset for plotting
                 end
                 handles.ch_data{ch_n}.sp_index = uint8(handles.ch_data{ch_n}.sp_index);
                 plot_counter = plot_counter+1;
            else
                max_cls = length(classes);
                handles.ch_data{ch_n}.index = zeros(max_cls,lts);
                handles.ch_data{ch_n}.fr = zeros(max_cls,ceil(lts/2));  %decimate 2
                handles.ch_data{ch_n}.frmax =zeros(max_cls,1);
                if handles.par.show_mav
                    handles.ch_data{ch_n}.mav = zeros(max_cls,lts);
                end
                for cl_n = 1:max_cls
                    class =  classes(cl_n);
                    sorted_times = cluster_class(cluster_class(:,1)==class,2);
                    sorted_times = sorted_times(sorted_times>(handles.par.tbeg*1000))-handles.par.tbeg*1000;
                    sorted_samples = ceil(sorted_times*sr/1e3);
                    sorted_samples = sorted_samples(sorted_samples<=lts);
                    handles.ch_data{ch_n}.index(cl_n,ceil(sorted_times*sr/1e3)) = 1;
                    if handles.par.show_fr
                        n_spike_timeline = length(handles.ch_data{ch_n}.index(cl_n,:));
                        integ_timeline_stim = conv(handles.ch_data{ch_n}.index(cl_n,:), int_window);
                        fr = integ_timeline_stim(round(half_width_gauss/sample_period)+1:2:n_spike_timeline+round(half_width_gauss/sample_period));
                        handles.ch_data{ch_n}.frmax(cl_n) = max(fr);
                        handles.ch_data{ch_n}.fr(cl_n,:) = single(fr /handles.ch_data{ch_n}.frmax(cl_n)+(plot_counter)); %decimate 2
                    end
                    if handles.par.show_mav
                        mav = conv(handles.ch_data{ch_n}.index(cl_n,:), ones(1,ceil(handles.par.window_len_mav*handles.sr)))/handles.par.window_len_mav;
                        handles.ch_data{ch_n}.mav(cl_n,:) = single(mav(1:lts)); % rescale 0 to 1, and add offset for plotting
                    end
                    plot_counter = plot_counter+1;
                end
            end
        end
    else
        set(handles.fr_axes,'Visible','off')
    end
    if handles.par.show_lfp
        plot_counter = plot_counter + max_channels;
		
		% target_res = 2*768;

		% % "raw"
		% fpass_l=2;
		% fpass_h=512;
		% fstop_l=0.5;
		% fstop_h=1000;

		% % theta
		% fpass_l = 4;
		% fpass_h = 8;
		% fstop_l = 3;
		% fstop_h = 15;

		% Rp=0.07;
		% Rs=20;

		% dec = round(sr/target_res);  
		% sr_LFP = sr/dec;

		% [orden_pass, Wnorm_pass] = ellipord([fpass_l*2/sr fpass_h*2/sr],[fstop_l*2/sr fstop_h*2/sr],Rp,Rs);
		% [z_pass,p_pass,k_pass] = ellip(orden_pass,Rp,Rs,Wnorm_pass);
		% [s_pass,g_pass] = zp2sos(z_pass,p_pass,k_pass);

		% tsmin = [TimeStamps(1) TimeStamps(floor(lts/4)+1) TimeStamps(floor(lts/2)+1) TimeStamps(floor(3*lts/4)+1)];  %check that it always start with TimeStamps(1) and nothing is missing between the pieces
		% tsmax = [TimeStamps(floor(lts/4)) TimeStamps(floor(lts/2)) TimeStamps(floor(3*lts/4)) TimeStamps(end)];

		% lfp=[];

		% for j=1:length(tsmin)
			% tinfind=find(TimeStamps>=tsmin(j),1);
			
			% tsupind=find(TimeStamps>tsmax(j),1);   % this will read until tsmax(j) or the closest to -inf
			% if isempty(tsupind)
				% tsupind = length(TimeStamps)+1;
			% end
			
			% filename=sprintf('NSX%d.NC5',handles.par.channels);
		    % f=fopen(fullfile(handles.par.folder_base,filename),'r','l');
			% fseek(f,(handles.min_record-1)*2,'bof');
			% samples_2_play = floor(sr * handles.par.rec_length);
			% x = fread(f1,samples_2_play,'int16=>double');
			
			% xf=filtfilt(s_pass,g_pass,x);
			% if with_notch
				% for kk=1:1:floor(fpass_h/50)
					% wo = 50*kk/(sr/2);  bw = wo/100; %bw = wo/144; %bw = wo/35;
					% [bnotch,anotch] = iirnotch(wo,bw);
					% xf=filtfilt(bnotch,anotch,xf);
				% end
			% end
			
			% xd = xf(1:dec:end);
			% lfp =[lfp xd];
		% end

		% % h_LFP = hilbert(lfp);
		% % h_power = abs(h_LFP).^2;
		% % h_angle=angle(h_LFP);
		
    end
    %set(handles.fr_axes,'YTick',0.5:1:plot_counter);
    
    
    handles.plot_counter = plot_counter;
    if plot_counter ~= 0
        
        ylim(handles.fr_axes,[0 plot_counter]);
        ylim(handles.axes_labels,[0 plot_counter]);
        xlim(handles.axes_labels,[-1 1])
        yborders = 0:plot_counter;
        plot(handles.axes_labels,[ones(1, length(yborders))*-1; ones(1, length(yborders))],[yborders;yborders],'-','color',[0,0,0]+0.6);
        text_n = 1;
        handles.clabels =cell(0,0);
        for ch_n = 1: max_channels
            ch = handles.par.channels(ch_n);
            classes = handles.par.classes{ch_n};
            
            if strcmp(classes,'mu')
                label = sprintf('Ch: %d \nmu',handles.par.channels(ch_n));
                handles.clabels{end+1,1} = label;
                handles.clabels{end,2} = text(0.1,-0.5+text_n,label,'Parent',handles.axes_labels, 'FontSize',11,'HorizontalAlignment','center','VerticalAlignment','middle','Rotation',90);
                if handles.par.show_fr
                    label = num2str(handles.ch_data{ch_n}.frmax,'%3.2f');
                    text(0.5,text_n-0.08,label,'Parent',handles.axes_labels, 'FontSize',9,'HorizontalAlignment','right','VerticalAlignment','cap')
                end
                text_n = text_n + 1;
            else
                for cl_n = 1:length(classes)
                    label = sprintf('Ch: %d \nCl: %d',handles.par.channels(ch_n),classes(cl_n));
                    handles.clabels{end+1,1} = label;
                    handles.clabels{end,2} = text(0.1,-0.5+text_n,label,'Parent',handles.axes_labels, 'FontSize',11,'HorizontalAlignment','center','VerticalAlignment','middle','Rotation',90);
                    if handles.par.show_fr
                        label = num2str(handles.ch_data{ch_n}.frmax(cl_n),'%3.2f');
                        text(0.5,text_n-0.08,label,'Parent',handles.axes_labels, 'FontSize',8,'HorizontalAlignment','right','VerticalAlignment','cap')
                    end
                    text_n = text_n + 1;
                end
            end
        end
        % if handles.par.show_lfp
            % labels{end} = 'LFP';
        % end
        set(handles.axes_labels,'Visible','off')
    end

    if handles.par.mode == 'a'
        set(handles.ref_label,'String','Current time:');
        handles.ev_time = -1 ;

        handles.audio_ref = 0;
        if exist(fullfile(handles.par.folder_resus,'finalevents_audio.mat'),'file')==2
            L=load(fullfile(handles.par.folder_resus,'finalevents_audio.mat'));
            if ~strcmp(handles.session,L.session)
                error('check the session as there seems to be a discrepancy between the one loaded and the one you are trying to process')
            end
            handles.events = L.events;
        end
        
        samples_2_play = floor(sr * handles.par.rec_length);

        if isnumeric(handles.par.audio_channel)
            f1 = fopen(fullfile(handles.par.folder_base,audio_channel_file),'r','l');
            fseek(f1,(handles.min_record-1)*2,'bof');
            y = fread(f1,samples_2_play,'int16=>double');
            fclose(f1);
        else
            y = load(handles.par.audio_file,handles.par.audio_variable);
            y = y.(handles.par.audio_variable);
            y = reshape(y(handles.min_record:end),1,[]);
        end
        ymin = prctile(y,0.5);
        ymax = prctile(y,99.5);
        handles.audio = (y(:)-ymin)/(ymax-ymin)*2-1;
        handles.player = audioplayer(handles.audio(1:end),sr,16);
        timer_period = 0.05;
        play(handles.player);
        
    elseif handles.par.mode == 'v'
        if exist(fullfile(handles.par.folder_resus,'finalevents_video.mat'),'file')==2
            L=load(fullfile(handles.par.folder_resus,'finalevents_video.mat'));
            if ~strcmp(handles.session,L.session)
                error('check the session as there seems to be a discrepancy between the one loaded and the one you are trying to process')
            end
            handles.events = L.events;
        end
        set(handles.ref_label,'String','Current frame:');
        handles.ev_frame = -1 ;
        v_viewer('Visible','off');
        handles.view_fig = findobj( 0, 'type', 'figure', 'tag', 'v_viewer');
        handles.fig = hObject;
        handles.vidObj = VideoReader(handles.par.video_file);

        set(handles.view_fig,'units','pixels');
        set(handles.view_fig,'position',[0 0 handles.vidObj.Width handles.vidObj.Height]);
        movegui(handles.view_fig,'northwest');
        
        v_viewer('Visible','on');
        timer_period = ceil(1/handles.vidObj.FrameRate*1000)/1000;
        handles.v_fr = handles.vidObj.FrameRate;
        
        handles.max_vf = handles.vidObj.NumberOfFrames;
        handles. vf_segment= ceil(handles.par.frame_len*handles.vidObj.FrameRate);
        setappdata(hObject,'video_frame',0);
        h = guidata(handles.view_fig);
        handles.video_axes = h.video_axes;
        handles.video_pause_tb = h.pause_tb;
        set(h.pause_tb,'Value',0)
        
    else
        error('invalid par.mode')
    end
    
    
    handles.timer_plot_loop = timer('Name','plot_loop','TimerFcn',{@plot_loop,hObject},'Period',timer_period,'ExecutionMode','fixedRate');
    guidata(hObject, handles);
    start(handles.timer_plot_loop)
    pause_tb_Callback(handles.pause_tb,[],handles)

% --- Executes on button press in selec_pb.
function selec_pb_Callback(hObject, eventdata, handles)
% hObject    handle to selec_pb (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(hObject,'Enable','off');    
uitoolbar = findall(handles.uitoolbar1, '-property', 'Enable');
set( uitoolbar,'Enable', 'off')
if get(handles.pause_tb,'Value') == 0
    set(handles.pause_tb,'Value',1.0);  
    pause_tb_Callback(handles.pause_tb,[],handles);
end
axes(handles.multimedia_plot)
[time_selected, aux,button] = ginput(1);                                          %gets the mouse input
if button == 3
	set(hObject,'Enable','on');
    set( uitoolbar,'Enable', 'on')
    return
end
%set(handles.lineav, 'Xdata', [time_selected time_selected])
sr = handles.sr;
FRlength = handles.par.frame_len;

if handles.par.mode == 'a'
    stop(handles.player)
    handles.ev_time = time_selected;
    new_init = floor(time_selected*sr);
    handles.player = audioplayer(handles.audio(new_init-handles.min_record:end),sr,16);
    handles.audio_ref = new_init-handles.min_record;
    guidata(hObject, handles);
    plot_loop([], [],hObject);

else
    handles.ev_frame = ceil(time_selected * handles.v_fr)-1; %because when call the plot loop add one
    warning('selec_pb_Callback...checkear con inicio video, registro, etc')
    video_frame = handles.ev_frame-1;%ceil(rel_time * handles.v_fr)-1;
    setappdata(handles.av_gui,'video_frame',video_frame);
    guidata(hObject, handles);
    plot_loop([], [],handles.av_gui)
    pause_tb_Callback(handles.pause_tb,[],handles);
end
set(handles.add_mark_pb,'Enable','on');
set(hObject,'Enable','on');    
set( uitoolbar,'Enable', 'on')



function av_gui_keypressfcn(hObject, eventdata, handles)
 % determine the key that was pressed 
%     setappdata(hObject,'keyPres',0);
 flag = getappdata(hObject,'keyPres');
 if flag == 0
     setappdata(hObject,'keyPres',1); 
     keyPressed = eventdata.Key; 
     switch keyPressed
         case 'subtract'
             uipushtool3_ClickedCallback(hObject, eventdata, handles)
         case 'space'
            button_state = get(handles.pause_tb,'Value');
            if button_state == 1
                set(handles.pause_tb,'Value',0.0);
            else
                set(handles.pause_tb,'Value',1.0);    
            end
            pause_tb_Callback(handles.pause_tb,[],handles);
         case 'leftarrow'
             prev_pb_Callback(handles.prev_pb,[],handles);
         case 'rightarrow'
            next_pb_Callback(handles.next_pb,[],handles);
         case 'return'
             if strcmp(get(handles.add_mark_pb,'Enable'),'on')
                add_mark_pb_Callback(hObject, eventdata, handles)
             end
         otherwise
              keyPressed = keyPressed(end); %for read crrectly the numpad numbers
              if isstrprop(keyPressed,'digit')
                nevent = str2num(keyPressed);
                if nevent>0 && nevent <= length(get(handles.stim_lb,'String'))
                    set(handles.stim_lb,'Value',nevent);
                end
              end
           
     end
 end
% --- Executes during object creation, after setting all properties.
function stim_lb_CreateFcn(hObject, eventdata, handles)
% hObject    handle to stim_lb (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes on button press in save_pb.
function save_pb_Callback(hObject, eventdata, handles)
% hObject    handle to save_pb (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
events = handles.events;
session = handles.session;
colors = handles.colors;
if length(events) > 0
    [~,Ieve] = sort(events,1);
    events = events(Ieve(:,2),:);
end
if handles.par.mode == 'a'
    times_audio = events(:,2);
    save(fullfile(handles.par.folder_resus,'finalevents_audio.mat'),'events','session','times_audio','colors')
end
if handles.par.mode == 'v'
    times_video = events(:,2);
    save(fullfile(handles.par.folder_resus,'finalevents_video.mat'),'events','session','times_video','colors')
end

function stim_lb_Callback(hObject, eventdata, handles)
% --- Executes on button press in set_param_pb.

function set_param_pb_Callback(hObject, eventdata, handles)
% hObject    handle to set_param_pb (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% --- Executes on button press in set_param_button.
restart = set_par_ui();
if restart
    stop_av(handles);
    start_av(hObject);
end

% --- Executes on button press in prev_pb.
function prev_pb_Callback(hObject, eventdata, handles)
stop(handles.timer_plot_loop);
sr = handles.sr;
FRlength = handles.par.frame_len;
button_state = get(handles.pause_tb,'Value');


if handles.par.mode == 'a'
    stop(handles.player)
    if handles.current_frame == 1
        new_init = sr*FRlength*(handles.current_frame-1)+1;
        handles.current_frame = handles.current_frame-1;
        handles.player = audioplayer(handles.audio(new_init:end),sr,16);
        handles.audio_ref = new_init;
        if button_state == 1
            play(handles.player);
            pause(handles.player);
            guidata(hObject, handles);
            plot_loop([],[],hObject);
        else
            play(handles.player);
            guidata(hObject, handles);
            start(handles.timer_plot_loop);
        end
        return
    end
    new_init = sr*FRlength*(handles.current_frame-2)+1;
    handles.current_frame = handles.current_frame-2;
    handles.player = audioplayer(handles.audio(new_init:end),sr,16);
    handles.audio_ref = new_init;
    if button_state == 1
        play(handles.player)
        pause(handles.player);
        guidata(hObject, handles);
        plot_loop([],[],hObject);
    else
        play(handles.player);
        guidata(hObject, handles);
        start(handles.timer_plot_loop);
    end
    
else
     fig = findobj( 0, 'type', 'figure', 'tag', 'av_gui');
     video_frame = getappdata(fig,'video_frame');
     if handles.current_frame == 1
        handles.current_frame = handles.current_frame-1;
        video_frame = handles.vf_segment*handles.current_frame;
        setappdata(fig,'video_frame',video_frame);
        if button_state == 1
            guidata(hObject, handles);
            plot_loop([],[],hObject);
            pause_tb_Callback(handles.pause_tb,[],handles);
        else
            guidata(hObject, handles);
            start(handles.timer_plot_loop);
        end
        return
    end
    handles.current_frame = handles.current_frame-2;
    video_frame = handles.vf_segment*handles.current_frame;
    setappdata(fig,'video_frame',video_frame);
    if button_state == 1
        guidata(hObject, handles);
        plot_loop([],[],fig);
        pause_tb_Callback(handles.pause_tb,[],handles);
    else
        guidata(hObject, handles);
        start(handles.timer_plot_loop);
    end
end


set(handles.add_mark_pb,'Enable','off');

% --- Executes on button press in next_pb.
function next_pb_Callback(hObject, eventdata, handles)
if handles.current_frame >= handles.frame_max
    return
end
stop(handles.timer_plot_loop);
sr = handles.sr;
FRlength = handles.par.frame_len;
button_state = get(handles.pause_tb,'Value');
if handles.par.mode == 'a'
    stop(handles.player);
    new_init = sr*FRlength*handles.current_frame+1;
    handles.player = audioplayer(handles.audio(new_init:end),sr,16);
    handles.audio_ref = new_init;
    if button_state == 1
        play(handles.player)
        pause(handles.player)
        guidata(hObject, handles);
        plot_loop([], [],hObject)
        pause_tb_Callback(handles.pause_tb,[],handles);
    else
        play(handles.player);
        guidata(hObject, handles);
        start(handles.timer_plot_loop);
    end
else
    video_frame = handles.vf_segment*handles.current_frame+1;
    setappdata(handles.av_gui,'video_frame',video_frame);
    if button_state == 1
        guidata(hObject, handles);
        plot_loop([], [],handles.av_gui)
        pause_tb_Callback(handles.pause_tb,[],handles);
    else
        guidata(hObject, handles);
        start(handles.timer_plot_loop);
    end 
end
set(handles.add_mark_pb,'Enable','off');


function add_mark_pb_Callback(hObject, eventdata, handles)

events = handles.events;
session = handles.session;
ev = size(events,1)+1;
nevent = get(handles.stim_lb,'Value');
handles.events(ev,1) = nevent;
if handles.par.mode == 'a'
    handles.events(ev,2) = handles.ev_time*1e6; % in microsec from the beginning of the recording
else
     handles.events(ev,2) =  handles.ev_frame;
     handles.ev_time = (handles.ev_frame-1)/handles.v_fr;
end
plot(handles.multimedia_plot,[1  1]*handles.ev_time,[-1 1],'-.','linewidth',2,'color',handles.colors(mod(nevent,size(handles.colors,1)),:));
text(handles.ev_time,1,[num2str(nevent) ' '],'Parent',handles.multimedia_plot,'Color',handles.colors(mod(nevent,size(handles.colors,1)),:),'FontSize',12,'HorizontalAlignment','right','VerticalAlignment','top');
set(handles.add_mark_pb,'Enable','off');
guidata(hObject, handles);


% --- Executes on button press in select_event_pb.
function select_event_pb_Callback(hObject, eventdata, handles)
% hObject    handle to select_event_pb (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
uitoolbar = findall(handles.uitoolbar1, '-property', 'Enable');
set( uitoolbar,'Enable', 'off')
set(handles.pause_tb,'Value',1.0);    
pause_tb_Callback(handles.pause_tb,[],handles);
rect = getrect(handles.multimedia_plot);
if rect(3)==0 || rect(4) == 0
    if exist('handles.rectan','var')
        delete(handles.rectan);
    end
    handles.selected_events = [];
    set(uitoolbar,'Enable','on')
    return
end
handles.rectan = plot(handles.multimedia_plot,[rect(1) ,rect(3)+rect(1) ,rect(3)+rect(1),rect(1),rect(1)],[ rect(2),rect(2), rect(4)+rect(2), rect(4)+rect(2),rect(2)],':r','linewidth',2);

if handles.par.mode == 'a'
    etind = rect(1)*1e6;
    etend = (rect(1) + rect(3))*1e6;
else
    etind = floor(rect(1) * handles.v_fr)+1;
    etend = floor((rect(1) + rect(3)) * handles.v_fr)+1;
end
if ~isempty(handles.events)
    handles.selected_events = (handles.events(:,2) >= etind & handles.events(:,2) <= etend);
else
    handles.selected_events = [];
end
guidata(hObject, handles);
set(uitoolbar,'Enable','on')

% --- Executes on button press in delete_pb.
function delete_pb_Callback(hObject, eventdata, handles)
% hObject    handle to delete_pb (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if ~isempty(handles.selected_events)
    pause_value = get(handles.pause_tb,'Value');
    if pause_value == 0
        set(handles.pause_tb,'Value',1.0);    
        pause_tb_Callback(handles.pause_tb,[],handles);
    end
    %stop(handles.timer_plot_loop);
    delete(handles.rectan);
    drawnow
    deev = find(handles.selected_events);
    for i = 1:length(deev)
         e = deev(i);
         nevent = handles.events(e,1);
         if handles.par.mode == 'a'
            tevent = handles.events(e,2)/1e6;
         else
            tevent = (handles.events(e,2)-1)/ handles.v_fr;
         end
         plot(handles.multimedia_plot,[1  1]*tevent,[-1 1],'-.','linewidth',2,'color','w');
         text(tevent,1,[num2str(nevent) ' '],'Parent',handles.multimedia_plot,'Color','w','FontSize',12,'HorizontalAlignment','right','VerticalAlignment','top');
         if handles.par.show_fr || handles.par.show_raster || handles.par.show_lfp
             plot(handles.fr_axes,[1  1]*handles.events(e,2)/1e6,[0 handles.plot_counter],':','linewidth',2,'color','w');
             text(handles.events(e,2)/1e6,handles.plot_counter,[num2str(nevent) ' '],'Parent',handles.fr_axes,'Color','w','FontSize',12,'HorizontalAlignment','right','VerticalAlignment','top');
         end
    end
            
    handles.events = handles.events(~ handles.selected_events,:);
    handles.selected_events = [];
    guidata(hObject, handles)
    %start(handles.timer_plot_loop)
    if pause_value == 0
        set(handles.pause_tb,'Value',0);    
        pause_tb_Callback(handles.pause_tb,[],handles);
    end
end
    
% --------------------------------------------------------------------
function uipushtool3_ClickedCallback(hObject, eventdata, handles)
sr = handles.sr;
FRlength = handles.par.frame_len;
ind_beg = (floor(handles.current_frame-1)*FRlength*sr+1);
ind_end = ceil(handles.current_frame*FRlength*sr);
if ind_end > handles.lts
	ind_end = handles.lts;
end

xlim(handles.multimedia_plot,[handles.ejex_temp(ind_beg) handles.ejex_temp(ind_end)]);
ylim(handles.multimedia_plot,[-1 1])
if handles.par.show_fr || handles.par.show_raster || handles.par.show_lfp
    xlim(handles.fr_axes,[handles.ejex_temp(ind_beg) handles.ejex_temp(ind_end)]);
end
% --------------------------------------------------------------------
function uipushtool4_ClickedCallback(hObject, eventdata, handles)

pause_value = get(handles.pause_tb,'Value');

if pause_value == 0
    set(handles.pause_tb,'Value',1.0);    
    pause_tb_Callback(handles.pause_tb,[],handles);
end
rect = getrect(handles.multimedia_plot);
if rect(3)==0 || rect(4) == 0
    return
end

tind = rect(1);
tend = (rect(1) + rect(3));
xlim(handles.multimedia_plot,[tind tend]);
if handles.par.show_fr || handles.par.show_raster || handles.par.show_lfp
    xlim(handles.fr_axes,[tind tend]);
end

if pause_value == 0
    set(handles.pause_tb,'Value',0.0);    
    pause_tb_Callback(handles.pause_tb,[],handles);
end

function av_gui_WindowKeyReleaseFcn(hObject, eventdata, handles)
    setappdata(hObject,'keyPres',0); 

% --- Executes during object creation, after setting all properties.
function time_input_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function multimedia_plot_CreateFcn(hObject, eventdata, handles)
	hObject.Toolbar.Visible = 'off';
    
function fr_axes_CreateFcn(hObject, eventdata, handles)
	hObject.Toolbar.Visible = 'off';
    
% --- Executes on button press in go2time.
function time_input_Callback(hObject, eventdata, handles)
    jump = str2num(handles.time_input.String);
    if isempty(jump)
        set(handles.time_input,'String','input time (sec)');
        return
    end

% --- Executes on button press in go2time.
function go2time_Callback(hObject, eventdata, handles)
    jump = str2num(handles.time_input.String);
    if isempty(jump)
        return
    end
    if jump < handles.par.tbeg
        return
    end
    sr = handles.sr;
    FRlength = handles.par.frame_len;
    frame = floor((jump - handles.par.tbeg)/FRlength); %previous frame to update in plot_loop

    if frame >= handles.frame_max
        return
    end

    stop(handles.timer_plot_loop);
    pause_value = get(handles.pause_tb,'Value');
    if pause_value == 0
        set(handles.pause_tb,'Value',1.0);    
        pause_tb_Callback(handles.pause_tb,[],handles);
    end    
    if handles.par.mode == 'a'
        stop(handles.player)
        
        handles.audio_ref = sr*FRlength*(frame) +1;
        new_init = handles.audio_ref;
        handles.current_frame = frame;
        handles.player = audioplayer(handles.audio(new_init:end),sr,16);
        play(handles.player)
        pause(handles.player);
        guidata(hObject, handles);
        plot_loop([],[],hObject);
    else
        handles.current_frame = frame;
        video_frame = handles.vf_segment*frame+1;
        setappdata(fig,'video_frame',video_frame);
        guidata(hObject, handles);
        plot_loop([],[],fig);
        pause_tb_Callback(1,[],handles);
    end
    if pause_value == 0
        set(handles.pause_tb,'Value',0);    
        pause_tb_Callback(handles.pause_tb,[],handles);
    end 
    set(handles.add_mark_pb,'Enable','off');


% --- Executes on button press in pushbutton12.
function create_rasters(hObject, eventdata, handles)
% hObject    handle to pushbutton12 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
    warning('not implemented')
