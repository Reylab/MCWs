function varargout = set_par_ui(varargin)
% SET_PAR_UI MATLAB code for set_par_ui.fig
%      SET_PAR_UI, by itself, creates a new SET_PAR_UI or raises the existing
%      singleton*.
%
%      H = SET_PAR_UI returns the handle to a new SET_PAR_UI or the handle to
%      the existing singleton*.
%
%      SET_PAR_UI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in SET_PAR_UI.M with the given input arguments.
%
%      SET_PAR_UI('Property','Value',...) creates a new SET_PAR_UI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before set_par_ui_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to set_par_ui_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help set_par_ui

% Last Modified by GUIDE v2.5 18-Nov-2015 00:46:47

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @set_par_ui_OpeningFcn, ...
                   'gui_OutputFcn',  @set_par_ui_OutputFcn, ...
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


% --- Executes just before set_par_ui is made visible.
function set_par_ui_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to set_par_ui (see VARARGIN)

% Choose default command line output for set_par_ui
handles.output = hObject;

handles.restart = false;
% Update handles structure
guidata(hObject, handles);


% this_folder = dir();
% if ~ismember('set_parameters.m',{this_folder.name})
%      copyfile(, [pwd filesep 'set_parameters.m']);
% end
%edit([pwd filesep 'set_parameters.m'])
text = fileread([fileparts(mfilename('fullpath')) filesep 'par_av_gui.m']);
set(handles.text_editor,'string',text);
% UIWAIT makes set_par_ui wait for user response (see UIRESUME)
uiwait(handles.set_par_ui);


% --- Outputs from this function are returned to the command line.
function varargout = set_par_ui_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.restart;
delete(handles.set_par_ui);

% 
% % --- Executes on button press in default.
% function default_Callback(hObject, eventdata, handles)
% % hObject    handle to default (see GCBO)
% % eventdata  reserved - to be defined in a future version of MATLAB
% % handles    structure with handles and user data (see GUIDATA)
% text = fileread([fileparts(mfilename('fullpath')) filesep 'set_parameters_DEFAULT.m']);
% set(handles.text_editor,'string',text);
% 

function text_editor_Callback(hObject, eventdata, handles)
% hObject    handle to text_editor (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of text_editor as text
%        str2double(get(hObject,'String')) returns contents of text_editor as a double


% --- Executes during object creation, after setting all properties.
function text_editor_CreateFcn(hObject, eventdata, handles)
% hObject    handle to text_editor (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
set(hObject, 'Max', 2); %// Enable multi-line string input to the editbox



% --- Executes on button press in save.
function save_Callback(hObject, eventdata, handles)
% hObject    handle to save (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
text = get(handles.text_editor, 'String');
fout = fopen([fileparts(mfilename('fullpath')) filesep 'par_av_gui.m'],'w');

for row = 1:size(text,1)
    fprintf(fout, '%s \n', strtrim(text(row,1:end)));
end

fclose(fout);
handles.restart = true;
% Update handles structure
guidata(hObject, handles);
set_par_ui_CloseRequestFcn(handles.set_par_ui, eventdata, handles)

% --- Executes when user attempts to close set_par_ui.
function set_par_ui_CloseRequestFcn(hObject, eventdata, handles)
% hObject    handle to set_par_ui (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if isequal(get(hObject, 'waitstatus'), 'waiting')
    % The GUI is still in UIWAIT, us UIRESUME
    uiresume(hObject);
else
    % The GUI is no longer waiting, just close it
    delete(hObject);
end
