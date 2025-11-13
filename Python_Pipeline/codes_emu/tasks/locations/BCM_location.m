%location file for Baylor College of Medicine
function params=BCM_location(location)
    params = struct;
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


    params.beh_machine = 1;
    params.proccesing_machine = 2;
    params.lang = 'english';
    params.device_resp='keyboard';
    params.keyboards = {0};
    params.remote_disk_root = 'Z:';
    params.use_photodiodo = true;
    params.lang = 'english';
    params.device_resp='gamepad';
    params.which_nsp_comment = 1;
    params.which_nsp_micro = 2;
    params.acq_network = true;
    %       params.additional_pics = 'pics_USA';
    params.additional_pics = 'pics_now';
    params.use_daq = true;

    params.acq_remote_folder_in_beh = '/media/acq';
    params.with_acq_folder = true; %raw data folder accessible from beh and processing
    params.acq_is_beh = false;
    params.acq_is_processing = false;
    
    params.backup_in_processing = '/mnt/acq-hdd/'; %path, if empty don't backup after ending
    params.acq_remote_folder_in_processing = '/media/acq';
    params.with_acq_folder = true;
    params.acq_is_processing = false;
    
    
    params.pics_root_beh = 'C:\Users\EMU - Behavior\Documents\MATLAB\Behavioral Tasks\HR\experimental_files\pics'; 
    params.pics_root_processing = 'C:\Users\EMU - Behavior\Documents\MATLAB\Behavioral Tasks\HR\experimental_files\pics'; %same as beh

    params.trellis_data_path =  'C:\Users\emuca\Trellis\datafiles'; %for ripple

    params.central_data_path = ''; %for blackrock including fileseparator
    params.root_beh = 'C:\Users\EMU - Behavior\Documents\MATLAB\Behavioral Tasks\HR\experimental_files';
    params.root_processing = 'C:\Users\EMU - Behavior\Documents\MATLAB\Behavioral Tasks\HR\experimental_files';  

    params.mapfile = '*.map';
    params.processing_rec_metadata = 'C:\Users\EMU - Behavior\Documents\MATLAB\Behavioral Tasks\HR\experimental_files\rec_metadata'; %mapfile, preprocessing, templates, etc
    params.beh_rec_metadata = params.processing_rec_metadata;

    params.early_figures_full_path = 'C:\Users\EMU - Behavior\Documents\MATLAB\Behavioral Tasks\HR\early_figures';


    if contains(location, '-RIP')
        params.system = 'RIP';
        params.use_BRK_comment = false;
        params.online_notches = true;
        params.additional_paths = {'C:\Users\EMU - Behavior\Documents\MATLAB\Behavioral Tasks\HR\JoyMEX', params.xippmex_path, params.cbmex_path}; %maybe params.cbmex_path
        params.ttl_device = 'MC';
    elseif contains(location, '-BRK')
        params.use_BRK_comment = true;
        params.online_notches = false;
        params.system = 'BRK';
        params.ttl_device = 'LJ';
        params.additional_paths = {'C:\Users\EMU - Behavior\Documents\MATLAB\Behavioral Tasks\HR\JoyMEX', params.cbmex_path, 'C:\Users\EMU - Behavior\Documents\MATLAB\Useful Codes\For use with BlackRock'};
    else
        error('System not found')
    end
end

      