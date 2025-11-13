function varargout = cx_lfp(varargin)
% cx_lfp MATLAB code for cx_lfp.fig
%      cx_lfp, by itself, creates a new cx_lfp or raises the existing
%      singleton*.
%
%      H = cx_lfp returns the handle to a new cx_lfp or the handle to
%      the existing singleton*.
%
%      cx_lfp('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in cx_lfp.M with the given input arguments.
%
%      cx_lfp('Property','Value',...) creates a new cx_lfp or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before cx_lfp_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to cx_lfp_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help cx_lfp

% Last Modified by GUIDE v2.5 14-Jun-2023 10:27:50

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @cx_lfp_OpeningFcn, ...
    'gui_OutputFcn',  @cx_lfp_OutputFcn, ...
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
end

% --- Executes just before cx_lfp is made visible.
function cx_lfp_OpeningFcn(hObject, ~, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to cx_lfp (see VARARGIN)

% Choose default command line output for cx_lfp
handles.output = hObject;
% Update handles structure
guidata(hObject, handles);
start_adq(hObject)
end

% --- Outputs from this function are returned to the command line.
function varargout = cx_lfp_OutputFcn(~, ~, handles)
if ~isempty(handles)
    varargout{1} = handles.output;
    fig = gcf;
    fig.WindowState='maximized';
end
end

function display_channel(n,h)
N = getappdata(h.cbmex_lfp,'N');

set(h.channels_lb,'Value',n);
ch_info = getappdata(h.cbmex_lfp,'ch_info');
par = getappdata(h.cbmex_lfp,'par');
ch_data = getappdata(h.cbmex_lfp,'ch_data');
if contains(ch_info.label{n},'AF01')
    idx_airflow2 = find(contains(ch_info.label,'AF02'));
    if ~isempty(idx_airflow2)
        ch_data(n).cont = ch_data(n).cont-ch_data(idx_airflow2).cont;
        ch_data(n).cont_filtered = ch_data(n).cont_filtered-ch_data(idx_airflow2).cont_filtered;

        ch_info.label{n} = 'AF01 - AF02';
    else
        warning(' "Airflow2" Not Found');
    end
end
if strcmp(ch_info.unit{n},'uV') && par.custom_filter.(ch_info.parsr{n}).enable
    additional_curve = 'Notches and Custom Filter';
else
    additional_curve = 'Notches';
end
ch_aux_label = sprintf('%s | ID: %d', ch_info.label{n},ch_info.ch(n));
process_info = getappdata(h.cbmex_lfp,'proccess_info');
inds_freqs_search_notch = getappdata(h.cbmex_lfp,'inds_freqs_search_notch');
custom_notches = [];
if ~isempty(process_info)
    chID = (par.instance>0)*(par.instance+1)*1000 + ch_info.ch(n);
    pi = find(cellfun(@(x) x==chID,{process_info.chID}));
    if ~isempty(pi)
        custom_notches = process_info(pi).notches.freq;
        ch_aux_label = [ch_aux_label sprintf(' (%d notches)',length(custom_notches))];
    end
end
set(h.channel_label,'String',ch_aux_label)

%set gui params
set(h.info_channel,'String',{sprintf('Sampl. Rate: %1.0f Hz',ch_info.sr(n)),...
    sprintf('Cont. Acq.: %s',ch_info.smpfilter{n})});
set(h.freqpanel,'Title',sprintf('Parameters for %1.0f Hz',ch_info.sr(n)))
pars = ch_info.parsr{n};
set(h.xpower_min,'String',num2str(par.x_power_manual.(pars).min));
set(h.xpower_max,'String',num2str(par.x_power_manual.(pars).max));
set(h.xpower_z_min,'String',num2str(par.x_power_manual.(pars).min_zoom));
set(h.xpower_z_max,'String',num2str(par.x_power_manual.(pars).max_zoom));

gui_par = [ch_info.parsr{n}  ch_info.unit{n}];

ylim_el = {'ypower_min','ypower_max','ypower_z_min','ypower_z_max','yraw_min','yraw_max','yfiltered_min','yfiltered_max'};
ylim_cv = {'fix_ypower_z_cb','fix_ypower_cb','fix_yraw_cb','fix_yfiltered_cb'};
for ed  = ylim_el
    set(h.(ed{1}),'String',num2str(getappdata(h.(ed{1}),gui_par)))
end
for cb  = ylim_cv
    v = getappdata(h.(cb{1}),gui_par);
    set(h.(cb{1}),'Value',v);
    enable_editlines(h.(cb{1}),v,h)
end
%spectrum
cla(h.spectrum);
sr_data = getappdata(h.cbmex_lfp,'sr_data');
N_label = sprintf('%d (%1.1fs)',N,sr_data(ch_info.sri(n)).fft_n_s / sr_data(ch_info.sri(n)).sr);

set(h.N_lb,'String',N_label);
line(h.spectrum,sr_data(ch_info.sri(n)).fs,ch_data(n).log_psd,'LineWidth',1.5);

if getappdata(h.fix_ypower_cb,gui_par)
    yl = [getappdata(h.ypower_min,gui_par),getappdata(h.ypower_max,gui_par)];
else
    ylim(h.spectrum,'auto');
    yl = ylim(h.spectrum);
    yl(1) = yl(1) - 0.05*(yl(2)-yl(1));
end
line(h.spectrum,sr_data(ch_info.sri(n)).fs,ch_data(n).log_psd_filtered,'Color','r','LineWidth',0.75)

for i = 1:length(custom_notches)
    line(h.spectrum,custom_notches(i)*[1 1],yl,'Color','k','LineWidth',0.5,'LineStyle','--', 'DisplayName', '')
end
if ~isempty(process_info) && ~isempty(pi)
    line(h.spectrum,sr_data(ch_info.sri(n)).fs(inds_freqs_search_notch),process_info(pi).notches.thr_db,'Color','m','LineWidth',0.7)
end
legend(h.spectrum,'Raw',additional_curve)
if ~isempty(custom_notches)
    chH = get(h.spectrum,'Children')';
    set(h.spectrum,'Children',[chH(end-1);chH(end);chH(1:end-2)']);
end
ylim(h.spectrum,yl)
xlim(h.spectrum,[par.x_power_manual.(pars).min,par.x_power_manual.(pars).max]);

%spectrum_zoom
cla(h.spectrum_zoom);
line(h.spectrum_zoom,sr_data(ch_info.sri(n)).fs,ch_data(n).log_psd,'LineWidth',1.5);

if getappdata(h.fix_ypower_z_cb,gui_par)
    yl = [getappdata(h.ypower_z_min,gui_par),getappdata(h.ypower_z_max,gui_par)];
else
    ylim(h.spectrum_zoom,'auto');
    yl = ylim(h.spectrum_zoom);
    yl(1) = yl(1) - 0.05*(yl(2)-yl(1));
end
line(h.spectrum_zoom,sr_data(ch_info.sri(n)).fs,ch_data(n).log_psd_filtered,'Color','r','LineWidth',0.75);

for i = 1:length(custom_notches)
    line(h.spectrum_zoom,custom_notches(i)*[1 1],yl,'Color','k','LineWidth',0.5,'LineStyle','--')
end
ylim(h.spectrum_zoom,yl)
xlim(h.spectrum_zoom,[par.x_power_manual.(pars).min_zoom,par.x_power_manual.(pars).max_zoom]);
part_line = (mod(N-1,par.n_blocks)+1)*sr_data(ch_info.sri(n)).t(end)/par.n_blocks;

%time_raw

ax = cla(h.time_raw);
ax.YAxis.Exponent = 0;
line(h.time_raw,sr_data(ch_info.sri(n)).t,ch_data(n).cont,'LineWidth',1.2);
xline(h.time_raw,part_line,'k','LineStyle','--','LineWidth',1.5);
ylabel(h.time_raw,['Raw (' ch_info.unit{n} ')'])
xlim(h.time_raw,[sr_data(ch_info.sri(n)).t(1) sr_data(ch_info.sri(n)).t(end)])
xticks(h.time_raw,linspace(sr_data(ch_info.sri(n)).t(1),sr_data(ch_info.sri(n)).t(end),par.n_blocks*2));

if getappdata(h.fix_yraw_cb,gui_par)
    ylim(h.time_raw,[getappdata(h.yraw_min,gui_par),getappdata(h.yraw_max,gui_par)])
end

%time_filtered
ax = cla(h.time_filtered);
ax.YAxis.Exponent = 0;
line(h.time_filtered,sr_data(ch_info.sri(n)).t,ch_data(n).cont_filtered,'Color','r','LineWidth',1.2);
xline(h.time_filtered,part_line,'k','LineStyle','--','LineWidth',1.5);
ylabel(h.time_filtered,{additional_curve,['(' ch_info.unit{n} ')']})

if getappdata(h.fix_yfiltered_cb,gui_par)
    ylim(h.time_filtered,[getappdata(h.yfiltered_min,gui_par),getappdata(h.yfiltered_max,gui_par)])
end
xlim(h.time_filtered,[sr_data(ch_info.sri(n)).t(1) sr_data(ch_info.sri(n)).t(end)])
xticks(h.time_filtered,linspace(sr_data(ch_info.sri(n)).t(1),sr_data(ch_info.sri(n)).t(end),par.n_blocks*2));
end


function buffer_loop(this_timer,~,cbmex_lfp,handles)
%function that is called using a timer and update the gui data
loop_time=tic();
par = getappdata(cbmex_lfp,'par');
stop(this_timer)
keep_loading = false;
buffer = getappdata(cbmex_lfp,'buffer');
matlab_losses = 0;
nsx_losses = 0;

try
    streams = device_com('get_stream');
    hpc = getappdata(cbmex_lfp, 'handle_plot_continuous');
    if hpc.isloading()
        hpc.update(streams)
    end

    loaded = zeros(length(par.channels),1);

    %copy streams to matlab buffer (just to the fist half, the one that is used later to plot)
    for ci = 1:length(par.channels)
        news = length(streams.data{ci});
        if (streams.lost_prev(ci)+streams.lost_post(ci)) > 0
            nsx_losses = 1 + nsx_losses;
        end
        loaded(ci) = max(min(news,buffer(ci).nmax-buffer(ci).nupdated),0);%min(news,buffer(ci).nmax*2-buffer(ci).nupdated);
        if loaded(ci)>0
            buffer(ci).data((buffer(ci).nupdated+1):(buffer(ci).nupdated+loaded(ci))) = streams.data{ci}(1:loaded(ci));
            buffer(ci).nupdated = loaded(ci) + buffer(ci).nupdated;
        end
        if buffer(ci).nmax > buffer(ci).nupdated
            keep_loading = true; %all have to be true
        end
    end

    N = getappdata(cbmex_lfp,'N');
    if keep_loading == false %all buffers are full
        sr_data = getappdata(cbmex_lfp,'sr_data');
        ch_info = getappdata(cbmex_lfp,'ch_info');
        ch_data = getappdata(cbmex_lfp,'ch_data');

        while(keep_loading==false)
            keep_loading = false;
            N = N + 1;
            Nover_1 = 1/N;
            bl_part = mod(N-1,par.n_blocks);

            for i = 1:length(par.channels)
                new_segment = (bl_part)*buffer(i).nmax+1:(bl_part+1)*buffer(i).nmax;
                ch_data(i).cont(new_segment) =  single(buffer(i).data(1:buffer(i).nmax))*ch_info.conversion(i);

                %copy extra samples to beggining of file and move nupdate
                buffer(i).data(1:buffer(i).nupdated - buffer(i).nmax) = buffer(i).data(buffer(i).nmax+1:buffer(i).nupdated);
                buffer(i).nupdated = buffer(i).nupdated - buffer(i).nmax;

                si = ch_info.sri(i) ;  %sample rate
                psd = periodogram(ch_data(i).cont(new_segment(1:sr_data(si).fft_n_s)),sr_data(si).win,sr_data(si).fft_n_s,sr_data(si).sr,'onesided');
                ch_data(i).psd(:) = (psd + ch_data(i).psd*(N-1))*Nover_1;
                if strcmp(ch_info.label{i},'AF01')
                    s = sr_data(si).custom_filter.SAF;
                    g = sr_data(si).custom_filter.GAF;
                elseif strcmp(ch_info.unit{i},'uV') && par.custom_filter.(ch_info.parsr{i}).enable
                    s = sr_data(si).custom_filter.S;
                    g = sr_data(si).custom_filter.G;
                else
                    s = sr_data(si).notch.S;
                    g = sr_data(si).notch.G;
                end
                ch_data(i).cont_filtered(new_segment)= fast_filtfilt(s,g,ch_data(i).cont(new_segment));
                psd = periodogram(ch_data(i).cont_filtered(new_segment(1:sr_data(si).fft_n_s)),sr_data(si).win,sr_data(si).fft_n_s,sr_data(si).sr,'onesided');
                ch_data(i).psd_filtered(:) = (psd + ch_data(i).psd_filtered*(N-1))*Nover_1;
                ch_data(i).log_psd(:) = 10*log10(ch_data(i).psd);
                ch_data(i).log_psd_filtered(:) = 10*log10(ch_data(i).psd_filtered);

                %update matlab buffer as before
                news = length(streams.data{i})-loaded(i);
                newloaded = max(min(news,buffer(i).nmax-buffer(i).nupdated),0);%min(news,buffer(ci).nmax*2-buffer(ci).nupdated);

                if newloaded>0
                    buffer(i).data(buffer(i).nupdated+1:buffer(i).nupdated+newloaded) = streams.data{i}((loaded(i)+1):(loaded(i)+newloaded));
                    buffer(i).nupdated = newloaded + buffer(i).nupdated;
                    loaded(i) = loaded(i) + newloaded;
                end
                if buffer(i).nmax > buffer(i).nupdated
                    keep_loading = true; %all have to be true
                end
            end
        end

        setappdata(cbmex_lfp,'ch_data',ch_data);
        setappdata(cbmex_lfp,'N',N)
        if bl_part==(par.n_blocks-1) && handles.stop_refresh_cb.Value==false
            display_channel(handles.channels_lb.Value,handles)
        end
    else
        if (getappdata(handles.calc_notches_pb,'waitting') == true) && N >= par.k_bartlett
            sr_data = getappdata(cbmex_lfp,'sr_data');
            ch_info = getappdata(cbmex_lfp,'ch_info');
            ch_data = getappdata(cbmex_lfp,'ch_data');
            u_electrodes = getappdata(handles.cbmex_lfp,'u_electrodes');
            cd_ID = arrayfun(@(x) x+(par.instance>0)*(par.instance+1)*1000,ch_info.ch(u_electrodes));
            if ~isempty(cd_ID)
                notches_folder = getappdata(handles.calc_notches_pb,'notches_folder');
                %                 calculate_notches(cd_ID,{ch_data(u_electrodes).psd},sr_data(ch_info.sri(u_electrodes(1))).fs,par.db_offset4thr,notches_folder);
                calculate_notches(cd_ID,{ch_data(u_electrodes).log_psd_filtered},sr_data(ch_info.sri(u_electrodes(1))).fs,par.db_offset4thr,notches_folder);
                pinfo = load([notches_folder filesep 'pre_processing_info.mat']);
                setappdata(cbmex_lfp,'proccess_info',pinfo.process_info)
                setappdata(cbmex_lfp,'inds_freqs_search_notch',pinfo.inds_freqs_search_notch)
            end
            button_done(handles.calc_notches_pb)
        elseif hpc.isready2plot()
            notches_folder = getappdata(handles.calc_notches_pb,'notches_folder');
            set(handles.plot_cont_pb,'BackgroundColor',[0.6, 0.9, 0.77]); %green
            hpc.notches_folder = notches_folder;
            hpc.start_plotting();
        end
    end

    %check if extra samples have to be copy to the matlab buffer
    for i = 1:length(par.channels)
        news = length(streams.data{ci})-loaded(ci);
        newloaded = max(min(news,buffer(ci).nmax*2-buffer(ci).nupdated),0);

        if newloaded>0
            buffer(ci).data(buffer(ci).nupdated+1:buffer(ci).nupdated+newloaded) = streams.data{ci}((loaded(ci)+1):(loaded(ci)+newloaded));
            buffer(ci).nupdated = newloaded + buffer(ci).nupdated;
            loaded(ci) = loaded(ci) + newloaded;
            if loaded(ci) < length(streams.data{ci})
                matlab_losses = matlab_losses + 1;
            end
        end
    end
    if hpc.isdone()
        if hpc.got_error()
            msgbox({hpc.get_error_str(),'details in command window'});
        end
        button_done(handles.plot_cont_pb)
        fprintf('PLOT CONTINUOUS DONE\n')
    end
catch ME
    warning(ME.message)
    save('debug_timer_Data','ME')
    error('error in timer loop saving all the variables!!!')
end
if nsx_losses>0
    warning('lossing continuos data in nsp buffer on %d channels.',nsx_losses)
end
if matlab_losses>0
    warning('lossing continuos data in Matlab buffer on %d channels.',matlab_losses)
end
setappdata(cbmex_lfp,'buffer',buffer)
this_timer.StartDelay = max(round(1.5-toc(loop_time),3),0.002);
start(this_timer)
end

function stop_refresh_cb_Callback(hObject, ~, ~)
return
end

% --- Executes on button press in restart_button.
function restart_button_Callback(hObject, ~, ~)
handles = guidata(hObject);
stop(handles.timer_buffer);
buffer = getappdata(handles.cbmex_lfp,'buffer');
ch_data = getappdata(handles.cbmex_lfp,'ch_data');
for c = 1:length(buffer)
    buffer(c).nupdated = 0;
    buffer(c).data(:) = 0;
    ch_data(c).cont(:) = 0;
    ch_data(c).cont_filtered(:) = 0;
    ch_data(c).psd(:) = 0;
    ch_data(c).psd_filtered(:) = 0;
    ch_data(c).log_psd(:) = 1;
    ch_data(c).log_psd_filtered(:) = 1;
end
set(handles.N_lb,'String','0');
setappdata(handles.cbmex_lfp,'N',0);
setappdata(handles.cbmex_lfp,'buffer',buffer);
setappdata(handles.cbmex_lfp,'ch_data',ch_data);
hpc = getappdata(handles.cbmex_lfp,'handle_plot_continuous');
hpc.reset();
display_channel(handles.channels_lb.Value,handles)
handles.timer_buffer.StartDelay = 0.001;
par = getappdata(handles.cbmex_lfp,'par');

button_reset(handles.calc_notches_pb)
if par.use_parallel
    button_reset(handles.plot_cont_pb)
end
if exist([pwd filesep 'pre_processing_info.mat'],'file')
    pinfo = load([pwd filesep 'pre_processing_info.mat']);
    proccess_info = pinfo.process_info;
    inds_freqs_search_notch = pinfo.inds_freqs_search_notch;
else
    proccess_info = [];
    inds_freqs_search_notch = [];
end
setappdata(handles.cbmex_lfp,'proccess_info',proccess_info);
setappdata(handles.cbmex_lfp,'inds_freqs_search_notch',inds_freqs_search_notch);
device_com('enable_chs',par.channels,false,[]);

pause(0.01)
start(handles.timer_buffer);
end

% --- Executes on button press in set_param_button.
function set_param_button_Callback(hObject, ~, ~)
stop_adq(hObject)
start_adq(hObject)
end

% --- Executes when user attempts to close cx_lfp.
function cx_lfp_CloseRequestFcn(hObject, ~, ~)
par = getappdata(hObject,'par');
try
    if ~isempty(par)
        stop_adq(hObject)
        if par.cbmex_close_ok
            device_com('close');
        end
        custompath = getappdata(hObject,'custompath');
        custompath.rm();
    end
end
delete(hObject);
end

function stop_adq(hObject)
handles = guidata(hObject);
try
    stop(handles.timer_buffer);
    delete(handles.timer_buffer);
catch
    warning('closed with errors')
end
guidata(hObject, handles);
end


function start_adq(hObject)
%     if exist('C:\Program Files\Blackrock Microsystems\NeuroPort Windows Suite', 'dir')
%         addpath('C:\Program Files\Blackrock Microsystems\NeuroPort Windows Suite')
%     elseif exist('C:\Program Files (x86)\Blackrock Microsystems\Cerebus Windows Suite', 'dir')
%         addpath('C:\Program Files (x86)\Blackrock Microsystems\Cerebus Windows Suite')
%     elseif exist('C:\Program Files (x86)\Blackrock Microsystems\NeuroPort Windows Suite', 'dir')
%         addpath('C:\Program Files (x86)\Blackrock Microsystems\NeuroPort Windows Suite')
%     else
%         warning('Blackrock Microsystems Path Not Found')
%     end

handles = guidata(hObject);

button_reset(handles.calc_notches_pb)
button_reset(handles.plot_cont_pb)

if isappdata(handles.cbmex_lfp,'par')
    par = getappdata(handles.cbmex_lfp,'par');
else
    par = par_cb_lfp();
    %default values not editable by the user:
    par.connection = [];
    par.instrument = [];
    par.open_msg = '';
end


addpath(fileparts(fileparts(mfilename('fullpath'))))
% custompath = reylab_custompath({'wave_clus_reylab', 'codes_for_analysis','mex','useful_functions','cx_lfp'});
custompath = reylab_custompath({'wave_clus_reylab', 'codes_for_analysis','mex','useful_functions'});
setappdata(handles.cbmex_lfp,'custompath',custompath);

app = cbmex_config(handles.cbmex_lfp,par);
waitfor(app)
par = getappdata(handles.cbmex_lfp,'par');
if isempty(par)
    disp('config window closed -> cx_lfp closed')
    cx_lfp_CloseRequestFcn(handles.cbmex_lfp)
    return
end
if ~exist('FiltFiltM','file')
    disp('Default filtfilt will be used.');
end
%create used_device info
par.used_device = par.devices(par.device_ix);
par.used_device.instance = par.instance;
if (par.use_parallel == true)
    pool = gcp('nocreate');
    if isempty(pool) % If already a pool, do not create new one.
        parpool();
    end
end
if par.use_parallel
    set(handles.plot_cont_pb,'Enable','on');
else
    set(handles.plot_cont_pb,'Enable','off');
end

handles.new_data_loaded = false;
chs = par.channels;

handles.cbmex_lfp.PaperPosition = handles.cbmex_lfp.Position;
handles.cbmex_lfp.PaperPositionMode = 'auto';

all_chs_info = getappdata(handles.cbmex_lfp,'all_chs_info');

chi = arrayfun(@(x) find(x==all_chs_info.ch), chs);
ch_info = struct;
for f=fieldnames(all_chs_info)'
    ch_info.(f{1})= all_chs_info.(f{1})(chi);
end
uniques_sr = unique(ch_info.sr);
ch_info.sri = arrayfun(@(x) find(uniques_sr==x),ch_info.sr);
setappdata(handles.cbmex_lfp,'ch_info',ch_info);
set(handles.channels_lb,'string',ch_info.label);

ch_data = struct();
buffer = struct();
sr_data = struct(); % selected using sr index
for si = 1:length(uniques_sr)
    sr = uniques_sr(si);
    sr_data(si).sr = sr;
    n_s = ceil(par.ffftlength*sr); %longuitude for raw data
    fft_n_s = 2^floor(log2(par.ffftlength*sr));
    nsamples = n_s*par.n_blocks;
    sr_data(si).n_s = n_s;
    sr_data(si).fft_n_s = fft_n_s;
    sr_data(si).win = barthannwin(fft_n_s);
    [~ ,fs] = periodogram(zeros(1,fft_n_s),sr_data(si).win,fft_n_s,sr_data(si).sr,'onesided');
    sr_data(si).fs = fs;
    sr_data(si).t = linspace(0,nsamples/sr,nsamples);

    %calc notchs for this freq
    %notchs = 1:min(par.num_notchs,floor((sr/2*0.9)/freqs_notch));
    K = 1; Z = []; P = [];

    for i= 1:length(par.freqs_notch)
        w = par.freqs_notch(i)/(sr/2);
        bw = par.notch_width/(sr/2);
        if w>=1 || w==0
            continue
        end
        [b_notch,a_notch] = iirnotch(w,bw);
        [zi,pi,ki] = tf2zpk(b_notch,a_notch);
        K = K * ki;
        Z(end+1:end+2) = zi;
        P(end+1:end+2) = pi;
    end

    [S,G] = zp2sos(Z,P,K);       % Convert to SOS
    parsr = ['f' num2str(sr)];
    sr_data(si).notch.S = S;
    sr_data(si).notch.G = G;

    if par.custom_filter.(parsr).enable
        Rp = par.custom_filter.(parsr).Rp;
        Rs = par.custom_filter.(parsr).Rs;
        wpassAF = [0.05*2/sr 5*2/sr];
        wpass = [par.custom_filter.(parsr).bp1*2/sr par.custom_filter.(parsr).bp2*2/sr];
        if strcmp(par.custom_filter.(parsr).filter_type,'ellip_stop_band')
            fstop = [par.custom_filter.(parsr).fstop1  ,par.custom_filter.(parsr).fstop2 ]*2/sr;
            [orden_pass, Wnorm_pass] = ellipord(wpass,fstop,Rp,Rs);
            [z_pass,p_pass,k_pass] = ellip(orden_pass,Rp,Rs,Wnorm_pass);
        else
            [z_pass,p_pass,k_pass] = ellip(par.custom_filter.(parsr).order,Rp,Rs,wpass);
        end
        fstopAF = [0.02  ,7]*2/sr;
        [orden_passAF, Wnorm_passAF] = ellipord(wpassAF,fstopAF,Rp,Rs);
        [z_passAF,p_passAF,k_passAF] = ellip(orden_passAF,Rp,Rs,Wnorm_passAF);

        k_pass = k_pass * K;
        z_pass = [z_pass; Z'];
        p_pass = [p_pass; P'];
        [s_pass,g_pass] = zp2sos(z_pass,p_pass,k_pass);
        [s_passAF,g_passAF] = zp2sos(z_passAF,p_passAF,k_passAF);
        sr_data(si).custom_filter.SAF = s_passAF;
        sr_data(si).custom_filter.GAF = g_passAF;        
        sr_data(si).custom_filter.S = s_pass;
        sr_data(si).custom_filter.G = g_pass;
    end
    sr_data(si).ci = [];
    for c = find(ch_info.sri==si)'
        sr_data(si).ci(end+1) = c;
        buffer(c).nmax = n_s;
        buffer(c).nupdated = 0;
        buffer(c).data = zeros(1,2*n_s,'int16');
        ch_data(c).cont = zeros(1,nsamples,'single');
        ch_data(c).cont_filtered = zeros(1,nsamples,'double');
        ch_data(c).psd = ones(length(fs),1,'double');
        ch_data(c).psd_filtered = ones(length(fs),1,'double');
        ch_data(c).log_psd = ones(length(fs),1,'double');
        ch_data(c).log_psd_filtered = ones(length(fs),1,'double');
    end
end
set(handles.N_lb,'String','0');
u_electrodes = find(arrayfun(@(x)strcmp('uV',x),ch_info.unit)' & arrayfun(@(x) sr_data(x).sr==30000,ch_info.sri));
setappdata(handles.cbmex_lfp,'u_electrodes',u_electrodes);
u_chID = (par.instance>0)*(par.instance+1)*1000 + [ch_info.ch(u_electrodes)];
u_labels = ch_info.label(u_electrodes);

notches_folder = getappdata(handles.calc_notches_pb,'notches_folder');
hpc = handle_plot_continuous(par.plot_cont_sec, par,...
    u_electrodes , u_chID, u_labels, ch_info.conversion(u_electrodes),notches_folder, handles.remove_artifacts_cb.Value);

setappdata(handles.cbmex_lfp,'handle_plot_continuous',hpc);
setappdata(handles.cbmex_lfp,'sr_data',sr_data);
setappdata(handles.cbmex_lfp,'buffer',buffer);
setappdata(handles.cbmex_lfp,'ch_data',ch_data);
setappdata(handles.cbmex_lfp,'par',par);
setappdata(handles.cbmex_lfp,'N',0);
%GUI params
if ~ isappdata(handles.fix_ypower_z_cb,'f3000uV')
    ylim_el = {'ypower','ypower_z','yraw','yfiltered'};
    ylim_cv = {'fix_ypower_z_cb','fix_ypower_cb','fix_yraw_cb','fix_yfiltered_cb'};
    for parstr = fieldnames(par.x_power_manual)'
        for unit = {'uV','mV'}
            field = [parstr{1} unit{1}];
            for cb = ylim_cv
                setappdata(handles.(cb{1}),field,false);
            end
            for et = ylim_el
                setappdata(handles.([et{1} '_min']),field,0);
                setappdata(handles.([et{1} '_max']),field,1);
            end
        end
    end
end
linkaxes([handles.time_raw,handles.time_filtered],'x')
handles.spectrum.XMinorGrid = 'on'; handles.spectrum.YMinorGrid = 'on';hold(handles.spectrum,'on');
handles.spectrum_zoom.XMinorGrid = 'on'; handles.spectrum_zoom.YMinorGrid = 'on';hold(handles.spectrum_zoom,'on');
xlabel(handles.spectrum,'Frequency (Hz)')
ylabel(handles.spectrum,'Power Spectrum (dB/Hz)')
xlabel(handles.time_filtered,'Time (sec)')
xlabel(handles.time_raw,'Time (sec)')
handles.time_raw.XMinorGrid = 'on'; handles.time_raw.YMinorGrid = 'on';
handles.time_filtered.XMinorGrid = 'on'; handles.time_filtered.YMinorGrid = 'on';
xticks(handles.time_raw,'manual'); xticks(handles.time_filtered,'manual')
xlabel(handles.spectrum_zoom,'Frequency (Hz)')
ylabel(handles.spectrum_zoom,'Power Spectrum (dB/Hz)')



handles.timer_buffer = timer('Name','buffer_timer','Period',100,...
    'ExecutionMode','fixedSpacing','StartDelay',0.02);
handles.timer_buffer.TimerFcn = {@buffer_loop,...
    handles.cbmex_lfp,handles};
%     if exist([pwd filesep 'pre_processing_info.mat'],'file')
%         pinfo = load([pwd filesep 'pre_processing_info.mat']);
%         proccess_info = pinfo.process_info;
%     else
%         proccess_info = [];
%     end
%     setappdata(handles.cbmex_lfp,'proccess_info',proccess_info)

if exist([pwd filesep 'pre_processing_info.mat'],'file')
    pinfo = load([pwd filesep 'pre_processing_info.mat']);
    proccess_info = pinfo.process_info;
    inds_freqs_search_notch = pinfo.inds_freqs_search_notch;
else
    proccess_info = [];
    inds_freqs_search_notch = [];
end
setappdata(handles.cbmex_lfp,'proccess_info',proccess_info);
setappdata(handles.cbmex_lfp,'inds_freqs_search_notch',inds_freqs_search_notch);

guidata(hObject, handles);
display_channel(1,handles)
device_com('enable_chs',par.channels,false,[]);
start(handles.timer_buffer)
end



% --- Executes on button press in prev_ch.
function change_ch_Callback(hObject, inc)
% hObject    handle to prev_ch (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles = guidata(hObject);
ch_num = handles.channels_lb.Value;
new_n = ch_num + inc;
maxch = size(handles.channels_lb.String,1);
if new_n > maxch
    new_n = 1;
elseif new_n == 0
    new_n = maxch;
end
display_channel(new_n,handles)
end

function save_b_Callback(hObject)
h = guidata(hObject);
ch_info = getappdata(h.cbmex_lfp,'ch_info');
ch_num = h.channels_lb.Value;
selpath = uigetdir('choose folder');
if (h.save_all_cb.Value ==0) && (h.save_micros_cb.Value ==0)
    saveas(h.cbmex_lfp,fullfile(selpath,[ch_info.label{ch_num} '.png']));
else
    maxch = size(h.channels_lb.String,1);
    oldvalue = h.stop_refresh_cb.Value;
    h.stop_refresh_cb.Value=true;
    if h.save_micros_cb.Value
        to_save = getappdata(h.cbmex_lfp,'u_electrodes');
    else
        to_save = 1:maxch;
    end
    for ch = circshift(1:maxch,maxch-h.channels_lb.Value)
        if ~any(to_save==ch)
            continue
        end
        display_channel(ch,h)
        ch_num = h.channels_lb.Value;
        drawnow
        saveas(h.cbmex_lfp,fullfile(selpath,[ch_info.label{ch_num} '.png']));
    end
    h.stop_refresh_cb.Value = oldvalue;
end
end

function channels_lb_Callback(hObject,eventdata,h)
display_channel(eventdata.Source.Value,h)
end


% --- Executes during object creation, after setting all properties.
function ax_CreateFcn(hObject)
a=axtoolbar(hObject,{'export','datacursor','pan'	,'zoomin','zoomout','restoreview'});
a.Visible = 'on';
end


% --- Executes on button press in fix_ypower_cb.
function fixscale_cb_Callback(hObject, ~)
% hObject    handle to fix_ypower_cb (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
h = guidata(hObject);
ch_info = getappdata(h.cbmex_lfp,'ch_info');
ch = h.channels_lb.Value;
field = [ch_info.parsr{ch}  ch_info.unit{ch}];
setappdata(hObject,field,hObject.Value);
enable_editlines(hObject.Tag,hObject.Value,h)
switch hObject.Tag
    case 'fix_ypower_cb'
        ax = h.spectrum;
        ymin = getappdata(h.ypower_min,field);
        ymax = getappdata(h.ypower_max,field);
    case 'fix_ypower_z_cb'
        ax = h.spectrum_zoom;
        ymin = getappdata(h.ypower_z_min,field);
        ymax = getappdata(h.ypower_z_max,field);
    case 'fix_yraw_cb'
        ax = h.time_raw;
        ymin = getappdata(h.yraw_min,field);
        ymax = getappdata(h.yraw_max,field);
    case 'fix_yfiltered_cb'
        ax = h.time_filtered;
        ymin = getappdata(h.yfiltered_min,field);
        ymax = getappdata(h.yfiltered_max,field);
end
if hObject.Value==1
    ylim(ax,[ymin ymax])
else
    ylim(ax,'auto')
end
end

function enable_editlines(checkbox_name,value,h)
if value==1
    enable = 'on';
else
    enable = 'off';
end
switch checkbox_name
    case 'fix_ypower_cb'
        set(h.ypower_min,'Enable',enable);
        set(h.ypower_max,'Enable',enable);
    case 'fix_ypower_z_cb'
        set(h.ypower_z_min,'Enable',enable);
        set(h.ypower_z_max,'Enable',enable);
    case 'fix_yraw_cb'
        set(h.yraw_min,'Enable',enable);
        set(h.yraw_max,'Enable',enable);
    case 'fix_yfiltered_cb'
        set(h.yfiltered_min,'Enable',enable);
        set(h.yfiltered_max,'Enable',enable);
end
end

function setlim_Callback(hObject, e, h)
new_value = str2num(e.Source.String);
ch_num = h.channels_lb.Value;
ch_info = getappdata(h.cbmex_lfp,'ch_info');
par = getappdata(h.cbmex_lfp,'par');
pars = ch_info.parsr{ch_num};

edit_tag = hObject.Tag;
gui_par = [pars  ch_info.unit{ch_num}];

if strcmp(edit_tag,'xpower_min') || strcmp(edit_tag,'xpower_max')
    if strcmp(edit_tag,'xpower_min')
        par.x_power_manual.(pars).min = new_value;
    else
        par.x_power_manual.(pars).max = new_value;
    end
    setappdata(h.cbmex_lfp,'par',par);
    xlim(h.spectrum,[par.x_power_manual.(pars).min,par.x_power_manual.(pars).max]);
end

if strcmp(edit_tag,'xpower_z_min') || strcmp(edit_tag,'xpower_z_max')
    if strcmp(edit_tag,'xpower_z_min')
        par.x_power_manual.(pars).min_zoom = new_value;
    else
        par.x_power_manual.(pars).max_zoom = new_value;
    end
    setappdata(h.cbmex_lfp,'par',par);
    xlim(h.spectrum_zoom,[par.x_power_manual.(pars).min_zoom,par.x_power_manual.(pars).max_zoom]);
end

if strcmp(edit_tag,'ypower_min') || strcmp(edit_tag,'ypower_max')
    if strcmp(edit_tag,'ypower_min')
        setappdata(h.ypower_min,gui_par,new_value);
    else
        setappdata(h.ypower_max,gui_par,new_value);
    end
    ylim(h.spectrum,[getappdata(h.ypower_min,gui_par),getappdata(h.ypower_max,gui_par)]);
end

if strcmp(edit_tag,'ypower_z_min') || strcmp(edit_tag,'ypower_z_max')
    if strcmp(edit_tag,'ypower_z_min')
        setappdata(h.ypower_z_min,gui_par,new_value);
    else
        setappdata(h.ypower_z_max,gui_par,new_value);
    end
    ylim(h.spectrum_zoom,[getappdata(h.ypower_z_min,gui_par),getappdata(h.ypower_z_max,gui_par)]);
end

if strcmp(edit_tag,'yraw_min') || strcmp(edit_tag,'yraw_max')
    if strcmp(edit_tag,'yraw_min')
        setappdata(h.yraw_min,gui_par,new_value);
    else
        setappdata(h.yraw_max,gui_par,new_value);
    end
    ylim(h.time_raw,[getappdata(h.yraw_min,gui_par),getappdata(h.yraw_max,gui_par)]);
end

if strcmp(edit_tag,'yfiltered_min') || strcmp(edit_tag,'yfiltered_max')
    if strcmp(edit_tag,'yfiltered_min')
        setappdata(h.yfiltered_min,gui_par,new_value);
    else
        setappdata(h.yfiltered_max,gui_par,new_value);
    end
    ylim(h.time_filtered,[getappdata(h.yfiltered_min,gui_par),getappdata(h.yfiltered_max,gui_par)]);
end
end


function plot_cont_pb_Callback(hObject, eventdata, handles)
selected_path = uigetdir('choose folder');
if isempty(selected_path) || all(selected_path==0)
    return
end
button_waiting(hObject)
setappdata(hObject,'waitting',true);
hpc = getappdata(handles.cbmex_lfp,'handle_plot_continuous');
hpc.start(selected_path);
par = getappdata(handles.cbmex_lfp,'par');
msgbox(sprintf('Channels will be plotted after acquiring  %.2f seconds of raw data.',par.plot_cont_sec));
end

function calc_notches_pb_Callback(hObject, eventdata, handles)
button_waiting(hObject)

selected_path = uigetdir('choose folder for pre_processing_info.mat file');
if isempty(selected_path) || all(selected_path==0)
    selected_path = pwd;
end
setappdata(hObject,'waitting',true);
setappdata(hObject,'notches_folder',selected_path);
par = getappdata(handles.cbmex_lfp,'par');
N = getappdata(handles.cbmex_lfp,'N');
if N<par.k_bartlett
    msgbox(sprintf('Notches will be computed when K_bartlett>=%d.',par.k_bartlett));
end
end

function button_waiting(hObject)
set(hObject,'enable','off');
set(hObject,'BackgroundColor',[1, 0.98, 0.702]); %yellow
end

function button_done(hObject)
set(hObject,'enable','on');
setappdata(hObject,'waitting',false);
set(hObject,'BackgroundColor',[0.94,0.94, 0.94]); %gray
end

function button_reset(hObject)
set(hObject,'enable','on');
set(hObject,'BackgroundColor',[0.94,0.94, 0.94]); %gray
setappdata(hObject,'waitting',false);
end


function save_micros_cb_Callback(hObject, eventdata, handles)
if get(hObject,'Value')==1
    set(handles.save_all_cb,'Value',0)
end
end


% --- Executes on button press in save_all_cb.
function save_all_cb_Callback(hObject, eventdata, handles)
if get(hObject,'Value')==1
    set(handles.save_micros_cb,'Value',0)
end
end

% --- Executes on button press in remove_artifacts_cb.
function remove_artifacts_cb_Callback(hObject, eventdata, handles)
%     if get(hObject,'Value')==1
%         set(handles.remove_artifacts_cb,'Value',0)
%     else
%         set(handles.remove_artifacts_cb,'Value',1)
%     end
end


% --- Executes on button press in pushbutton20.
function create_RIPnotches_file(hObject, eventdata, h)  % TERMINAR DE REVISAR (VER NOBRE FUNCION)
% hObject    handle to pushbutton20 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
MAX_NOTCHES = 30;
n = h.channels_lb.Value;
ch_info = getappdata(h.cbmex_lfp,'ch_info');
% par = getappdata(h.cbmex_lfp,'par');
% ch_data = getappdata(h.cbmex_lfp,'ch_data');
% if strcmp(ch_info.unit{n},'uV') && par.custom_filter.(ch_info.parsr{n}).enable
%     additional_curve = 'Notches and Custom Filter';
% else
%     additional_curve = 'Notches';
% end
% ch_aux_label = sprintf('%s | ID: %d', ch_info.label{n},ch_info.ch(n));
process_info = getappdata(h.cbmex_lfp,'proccess_info');
if ~isempty(process_info)
    notches_folder = getappdata(h.calc_notches_pb,'notches_folder');
    fileID = fopen(fullfile(notches_folder,'raw_out.coeff'),'w');
    chID = ch_info.ch(n);
    pi = find(cellfun(@(x) x==chID,{process_info.chID}));
    n_notches = process_info(pi).n_notches;
    notches_info = process_info(pi).notches;

    Z = []; P = []; K = 1;
    [~, ix_notch] = sort(notches_info.abs_db,'descend');
    n_notches = min(MAX_NOTCHES, length(notches_info.abs_db));

    fprintf(fileID,'%s %d\n','SOS',n_notches);
    fprintf(fileID,'\n');

    for ni = 1:n_notches
        zpix = ix_notch(ni)*2+(-1:0);
        Z(end+1:end+2) = notches_info.Z(zpix);
        P(end+1:end+2) = notches_info.P(zpix);
        K = K *notches_info.K(ix_notch(ni));
    end
    %     S = zp2sos(notches_info.Z,notches_info.P,prod(notches_info.K));       % Convert to SOS
    S = zp2sos(Z,P,K);       % Convert to SOS

    for ii=1:n_notches
        fprintf(fileID,' %0.14f %0.14f %0.14f %0.14f \n',S(ii,3),S(ii,2),S(ii,5),S(ii,6));
    end
    fclose(fileID);
else
    warning('there is no process_info with nocthes')
end

end