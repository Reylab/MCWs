function varargout = v_viewer(varargin)
% V_VIEWER MATLAB code for v_viewer.fig
%      V_VIEWER, by itself, creates a new V_VIEWER or raises the existing
%      singleton*.
%
%      H = V_VIEWER returns the handle to a new V_VIEWER or the handle to
%      the existing singleton*.
%
%      V_VIEWER('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in V_VIEWER.M with the given input arguments.
%
%      V_VIEWER('Property','Value',...) creates a new V_VIEWER or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before v_viewer_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to v_viewer_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help v_viewer

% Last Modified by GUIDE v2.5 22-Mar-2016 12:23:53

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @v_viewer_OpeningFcn, ...
                   'gui_OutputFcn',  @v_viewer_OutputFcn, ...
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


% --- Executes just before v_viewer is made visible.
function v_viewer_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to v_viewer (see VARARGIN)

% Choose default command line output for v_viewer
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes v_viewer wait for user response (see UIRESUME)
% uiwait(handles.v_viewer);


% --- Outputs from this function are returned to the command line.
function varargout = v_viewer_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;
handles.main_fig = findobj( 0, 'type', 'figure', 'tag', 'av_gui');
setappdata(handles.v_viewer,'shift_matrix',0);
guidata(hObject,handles);

% --- Executes on button press in pause_tb.
function pause_tb_Callback(hObject, eventdata, handles)
% hObject    handle to pause_tb (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
button_state = get(hObject,'Value');
h = guidata(handles.main_fig);
set(h.pause_tb,'Value',button_state);    
av_gui('pause_tb_Callback',h.pause_tb, [], h)

% --- Executes on button press in select_frame_pb.
function select_frame_pb_Callback(hObject, eventdata, handles)
% hObject    handle to select_frame_pb (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if get(handles.pause_tb,'Value') ==0
    return
end
[x, y,button] = ginput(1);                                          %gets the mouse input
if button == 3
	return
end
hm = guidata(handles.main_fig);
w = hm.vidObj.Width ;
h = hm.vidObj.Height;
video_frame = getappdata(hm.av_gui,'video_frame');
rel_fr = floor(y/h)*hm.par.nrows + ceil(x/w);
shift = getappdata(handles.v_viewer,'shift_matrix');
hm.ev_frame = rel_fr + video_frame-hm.par.fr_prev+shift*hm.par.fr_prev;
if hm.ev_frame <= 0
    return
end
guidata(handles.main_fig,hm);
set(hm.add_mark_pb,'Enable','on');
hold(handles.video_axes, 'on')
text(x,y,['Fr: ' num2str(hm.ev_frame)],'HorizontalAlignment','center','BackgroundColor','w','EdgeColor','r','Parent',handles.video_axes)
hold(handles.video_axes, 'off')


% --- Executes on key release with focus on v_viewer or any of its controls.
function v_viewer_WindowKeyReleaseFcn(hObject, eventdata, handles)
    setappdata(handles.main_fig,'keyPres',0); 


% --- Executes on key press with focus on v_viewer or any of its controls.
function v_viewer_WindowKeyPressFcn(hObject, eventdata, handles)
flag = getappdata(handles.main_fig,'keyPres');
if flag == 0
     keyPressed = eventdata.Key;
     switch keyPressed
         case 'leftarrow'
            setappdata(handles.main_fig,'keyPres',1);
            fr_prev_pb_Callback([],[],handles);
         case 'rightarrow'
            setappdata(handles.main_fig,'keyPres',1);
            fr_next_pb_Callback([],[],handles);
         otherwise
           av_gui('av_gui_keypressfcn',handles.main_fig,eventdata, guidata(handles.main_fig))
     end
end

 

% --- Executes on button press in fr_prev_pb.
function fr_prev_pb_Callback(hObject, eventdata, handles)
if get(handles.pause_tb,'Value') ==0
    return
end
av_gui('show_im_matrix',guidata(handles.main_fig),-1)

% --- Executes on button press in fr_next_pb.
function fr_next_pb_Callback(hObject, eventdata, handles)
if get(handles.pause_tb,'Value') ==0
    return
end
av_gui('show_im_matrix',guidata(handles.main_fig),1)
