function params = MCW_location(location)
params = struct;
%add -SS for single switch mode
%add -WPTB for open PTB in a window and not fullscreen
params.location = location;

if contains(location,'-WPTB')
    params.windowRect = [0,0,1920,1080]; %usefull to add a separaed window in a extended screen
end

if ispc
    params.ptb_priority_normal = 0;
    params.ptb_priority_high = 1;               
    paths = {'C:\Program Files\Blackrock Microsystems\NeuroPort Windows Suite',...
        'C:\Program Files (x86)\Blackrock Microsystems\Cerebus Windows Suite',...
        'C:\Program Files (x86)\Blackrock Microsystems\NeuroPort Windows Suite'};
    for path = paths
        if exist(path{1}, 'dir')
            params.cbmex_path = genpath(path{1});
            break
        end
    end
    valnames = winqueryreg('name','HKEY_LOCAL_MACHINE','SOFTWARE\Microsoft\Windows\CurrentVersion\Installer\Folders');
    xippmexix = find(cellfun(@(x) ~isempty(regexp(x,'(Ripple\\Trellis\\Tools\\xippmex\\)$','match')),valnames));
    if ~isempty(xippmexix)
        params.xippmex_path=valnames{xippmexix};
    end
end
if isunix
    params.ptb_priority_normal = 0;
    params.ptb_priority_high = 1;
    params.xippmex_path = '/opt/Trellis/Tools/xippmex';
end

params.additional_pics = fullfile('pics_space','pics_now');
% params.keyboards = {'Chicony  HP Wired Desktop 320K Keyboard','Microsoft Microsoft® 2.4GHz Transceiver v8.0'};
params.keyboards = {'Microsoft Microsoft® 2.4GHz Transceiver v8.0';'Logitech K850'};
params.lang = 'english';
params.beh_rec_metadata = '/home/user/share/experimental_files/rec_metadata'; %mapfile, preprocessing, templates, etc
params.acq_remote_folder_in_beh = '/media/acq';
params.acq_remote_folder_in_processing = '/media/acq';
params.with_acq_folder = true;
params.acq_is_beh = false;
params.acq_is_processing = false;
params.copy_backup = true;
params.backup_path = '/mnt/acq-hdd';
params.ttl_device = 'LJ'; % LJ or MC
params.use_photodiodo = true;
params.device_resp='gamepad';

%single mode
if strcmp(location, 'MCW-BEH-RIP')
      params.beh_machine = 1;
      params.proccesing_machine = 2;         
      params.trellis_data_path = 'C:\Users\user\Trellis\datafiles'; %including fileseparator
      params.additional_paths = {params.xippmex_path};
      params.use_daq = true;
      params.root_processing = '/home/user/share/experimental_files/';
      params.root_beh = '/home/user/share/experimental_files/';
      params.pics_root_beh = '/home/user/share/experimental_files/pics';
      params.pics_root_processing = '/home/user/share/experimental_files/pics';
      params.mapfile = '*.map';
      params.processing_rec_metadata = '/home/user/share/experimental_files/rec_metadata'; %mapfile, preprocessing, templates, etc
      params.early_figures_full_path = '/home/user/share/early_figures';
      params.system = 'RIP';
      params.online_notches = true;
      params.copy_backup = true;
      params.backup_path = '/home/user/share/experimental_files/';
      params.acq_network = true;
    return
end

if strcmp(location, 'MCW-ABTL-RIP')
      current_user = getenv('USER');    
      dir_base = sprintf('/home/%s',current_user); 
      params.beh_machine = 1;
      params.proccesing_machine = 2;         
      params.trellis_data_path = [dir_base '/Trellis/dataFiles']; %including fileseparator
      params.additional_paths = {params.xippmex_path};
      params.use_daq = true;
      params.root_processing = [dir_base '/ReyLab/experimental_files/'];
      params.root_beh = [dir_base '/ReyLab/experimental_files/'];
      params.pics_root_beh = [dir_base '/ReyLab/experimental_files/pics'];
      params.pics_root_processing = [dir_base '/ReyLab/experimental_files/pics'];
      params.mapfile = '*.map';
      params.processing_rec_metadata = [dir_base '/ReyLab/experimental_files/rec_metadata']; %mapfile, preprocessing, templates, etc
      params.early_figures_full_path = [dir_base '/ReyLab/early_figures'];
      params.codes_for_analysis = [dir_base '/Documents/GitHub/codes_emu/codes_for_analysis'];
      params.system = 'RIP';
      params.online_notches = true;
      params.copy_backup = true;
      params.backup_path = [dir_base '/ReyLab/experimental_files/'];
      params.ttl_device = 'MC';
      params.acq_network = true;
      params.acq_is_processing = true;
      params.beh_rec_metadata = [dir_base '/ReyLab/experimental_files/rec_metadata']; %mapfile, preprocessing, templates, etc
      params.acq_remote_folder_in_beh = [dir_base '/Trellis/dataFiles'];
      params.acq_remote_folder_in_processing = [dir_base '/Trellis/dataFiles'];
      params.with_acq_folder = true;
      params.acq_is_processing = true;
    return
end

if strcmp(location, 'MCW-ABT-RIP')
      params.beh_machine = 1;
      params.proccesing_machine = 2;         
      params.trellis_data_path = 'C:\Program Files (x86)\Ripple\Trellis\datafiles'; %including fileseparator
      params.additional_paths = {params.xippmex_path};
      params.use_daq = true;
      params.root_processing = 'C:\ReyLab\experimental_files\';
      params.root_beh = 'C:\ReyLab\experimental_files\';
      params.pics_root_beh = 'C:\ReyLab\experimental_files\pics';
      params.pics_root_processing = 'C:\ReyLab\experimental_files\pics';
      params.mapfile = '*.map';
      params.processing_rec_metadata = 'C:\ReyLab\experimental_files\rec_metadata'; %mapfile, preprocessing, templates, etc
      params.early_figures_full_path = 'C:\ReyLab\early_figures';
      params.system = 'RIP';
      params.online_notches = true;
      params.copy_backup = false;
      params.ttl_device = 'MC';
      params.beh_rec_metadata = 'C:\\ReyLab\\experimental_files\\rec_metadata'; %mapfile, preprocessing, templates, etc
      params.acq_remote_folder_in_beh = 'C:\Program Files (x86)\Ripple\Trellis\datafiles';
      params.acq_remote_folder_in_processing = 'C:\Program Files (x86)\Ripple\Trellis\datafiles';
      params.with_acq_folder = true;
      params.acq_is_processing = true;
    return
end

  params.trellis_data_path = 'C:\Users\user\Trellis\datafiles'; %including fileseparator
  params.acq_network = true;
  params.additional_paths = {params.xippmex_path};
  
  params.use_daq = true;
  params.root_processing = '/home/user/share/experimental_files/';
  params.root_beh = '/home/user/share/experimental_files/';
  params.pics_root_beh = '/home/user/share/experimental_files/pics';
  params.pics_root_processing = '/home/user/share/experimental_files/pics';
  params.mapfile = '*.map';
  params.processing_rec_metadata = '/home/user/share/experimental_files/rec_metadata'; %mapfile, preprocessing, templates, etc
  params.early_figures_full_path = '/home/user/share/early_figures';

if contains(location, '-SS')
  params.beh_machine = 'BEH-REYLAB';
  params.proccesing_machine = 'TOWER-REYLAB'; 
end
if contains(location, '-RIP')
%     params.offset=5;
%     params.offset=6;
    if ~contains(location, '-SS')
        params.beh_machine = '192.168.137.130'; %using BRK switch
        proccesing_machine_ips = {'192.168.137.226','192.168.137.228'};  %using BRK switch
        for i = 1: numel(proccesing_machine_ips)
            [ping_return,~] = system(['ping -c 1 -W 0.1 -q ' proccesing_machine_ips{i}]);
            if ping_return==0
                params.proccesing_machine = proccesing_machine_ips{i};
                break
            end
        end
    end
    params.system = 'RIP';
    params.online_notches = true;

elseif contains(location, '-BRK')
%    params.offset=5;
%    params.offset=6;
   params.which_nsp_micro = 1;
   if ~contains(location, '-SS')
       params.beh_machine = '192.168.42.130'; %using  ripple switch
       proccesing_machine_ips = {'192.168.42.226','192.168.42.228'}; %using ripple switch
       for i = 1: numel(proccesing_machine_ips)
           [ping_return,~] = system(['ping -c 1 -W 0.1 -q' proccesing_machine_ips{i}]);
           if ping_return==0
               params.proccesing_machine = proccesing_machine_ips{i};
               break
           end
       end
   end
   params.online_notches = false;
   params.system = 'BRK';
else
   error('device inside MCW not found')
end
end
