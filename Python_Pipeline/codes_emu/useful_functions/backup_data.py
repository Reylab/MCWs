import os
from os import system,path,mkdir 
import sys
from glob import glob
from shutil import move, rmtree, copy2
import progressbar
import shutil
from os.path import  join, exists, isfile



def sync_folder(names, source, dest):
    command = f"rsync -r --size-only --include='*/' --include='*/*' --exclude='*' --info=progress2 {source}/{names} {dest}"
    filenames = source + '/' + names
    folders_to_bk = glob(filenames)
    if len(folders_to_bk) == 0:
        print("No folders found with the names: " + filenames)
        return 0
    print("Running: " + command)

    total_files = sum(len(files) for _, _, files in os.walk(source))

    widgets = [
        'Progress: ', progressbar.Percentage(),
        ' ', progressbar.AdaptiveETA(),
        ' | Folders Synced: ', progressbar.Counter(format='%(value)d/%(max_value)d'),
        ' | Current Folder: ', progressbar.SimpleProgress(format='%(value)s'),
    ]

    progress_bar = progressbar.ProgressBar(widgets=widgets, max_value=total_files).start()

    def update_progress(count):
        progress_bar.update(count)

    synced_folders = set()

    def update_synced_folders(folder):
        synced_folders.add(folder)

    status = system(command + f" | tee /dev/tty | awk '/to-check=/ {{ print $1 }}' | awk -F'=' '{{ print $2 }}' | xargs -I % bash -c 'echo %/{total_files} | bc' | awk -F. '{{ print $1 }}' | awk 'NR>1' | xargs -I % bash -c 'update_progress %'")
    
    for folder in folders_to_bk:
        update_synced_folders(folder)

    progress_bar.finish()

    if status != 0:
        print("Error in command (read above)!!!!!!!")
    else:
        print("Sync completed successfully")
    
    print("Synced folders:")
    for folder in synced_folders:
        print(folder)
    
    return status

def check_mounted(folder):
    with open('/proc/mounts') as myfile:
        return folder in myfile.read()

def move_folder_to_transferred(names, acq_folder):
    transferred_folder =  acq_folder + '/' + 'transferred'
    if not path.exists(transferred_folder):
        mkdir(transferred_folder)
    folder_content =  glob(acq_folder+ '/' + names)
    for f in folder_content:
        if path.isdir(f):
            move(f, acq_folder + '/' + 'transferred')

def remove_folder_in_transferred(names, acq_folder,logfile):
    transferred_folder =  acq_folder + '/' + 'transferred'
    if not path.exists(transferred_folder):
        return
    folder_content =  glob(acq_folder+ '/transferred/' + names)
    with open(logfile, 'a') as f: 
        for folder in folder_content:
            if path.isdir(folder):
                f.write(folder+'\n')
                rmtree(folder)

if __name__ == '__main__':


    subject_id_file = '/home/user/scripts/current_subject_id.txt'
    server_folder = '/media/sEEG_DATA'
    acq_folder = '/media/acq'
    beh_folder = '/home/user/share'
    local_folder = '/mnt/acq-hdd'

    interactive = len(sys.argv)==1
    if not interactive:
        res = sys.argv[1]

    with open(subject_id_file, 'r') as f:
        subject_id = f.read().strip()

    names = 'EMU-*_subj-' + subject_id + '*'
    logfile = acq_folder + '/emulog_' + subject_id + '.txt'
    #check beh and acq in tower files
    # if not finish move to 
    if interactive:
        print('0) Check and finish BEH and ACQ/transferred copy to tower backups.')
        print('1) Backup ACQ files to tower backups.')
        print('2) Backup TOWER files to RACK and remove trasnferred in acq.')


        res_ok = False
        while(not res_ok):
            res = input('Choose an option number (close to quit): ')
            res_ok = res.isnumeric() and int(res)>=0 and int(res)<3

    l_backup = local_folder +'/' +  subject_id
    if not path.exists(l_backup):
        mkdir(l_backup)

    local_backup = local_folder +'/' +  subject_id + '/EMU' 
    if not path.exists(local_backup):
        mkdir(local_backup)

    if res=='0':
        #check if beh is mounted
        beh_mounted = check_mounted(beh_folder) 
        if beh_mounted:
     
            sync_folder(names, beh_folder +'/experimental_files', local_backup) #se tiene que borrar una vez que se hayan copiado (por ahora lo hacemos a mano despues de que se haga backup en rack)

            # Folders copied from beh to Local
            local_copied_folders = set()
            beh_folders = glob(beh_folder + '/experimental_files/' + names)
            for folder in beh_folders:
                if path.isdir(folder):
                    folder_name = os.path.basename(folder)
                    local_copied_folders.add(folder_name)

            acq_mounted = check_mounted(acq_folder)
            if acq_mounted:
                # Copy files from acq to local
                acq_files = glob(acq_folder + '/*')
                copied_files = []
                for file in acq_files:
                    if isfile(file):
                        file_name = os.path.basename(file)
                        folder_name = os.path.splitext(file_name)[0]
                        if any(item in folder_name for item in local_copied_folders):
                            folder_name = folder_name.replace('_RIP', '')
                            dest_folder = join(local_backup, folder_name)
                            copy2(file, dest_folder)
                            copied_files.append(file)

                if len(copied_files) > 0:
                    print("Synced files from acq_folder to local_folder:")
                    for file in copied_files:
                        print(file)
                else:
                    print("No files were found in acq_folder matching the folders.")

                # Move copied documents from acq to transferred_folder
                transferred_folder = acq_folder + '/' + 'transferred'
                if not path.exists(transferred_folder):
                    mkdir(transferred_folder)
                for file in copied_files:
                    move(file, transferred_folder)

                # Create a folder inside transferred with the name of the files
                for file in copied_files:
                    file_name = os.path.splitext(os.path.basename(file))[0]
                    transferred_subfolder = transferred_folder + '/' + file_name
                    if not path.exists(transferred_subfolder):
                        mkdir(transferred_subfolder)

                # Move the files to the corresponding folder within transferred
                for file in copied_files:
                    file_name = os.path.splitext(os.path.basename(file))[0]
                    transferred_subfolder = transferred_folder + '/' + file_name
                    move(transferred_folder + '/' + os.path.basename(file), transferred_subfolder + '/' + os.path.basename(file))

                # Move copied folders from beh to transferred_folder
                transferred_folder_beh = beh_folder + '/experimental_files' + '/' + 'transferred_beh'
                if not path.exists(transferred_folder_beh):
                    mkdir(transferred_folder_beh)
                beh_folders_to_move = glob(beh_folder + '/experimental_files/' + names)
                for folder in beh_folders_to_move:
                    if path.isdir(folder):
                        move(folder, transferred_folder_beh)
                
            else:
                print('Warning!: ACQ not mounted, unable to check copy there')
        else:
            print('Warning!: BEH not mounted, unable to check copy there')

    elif  res=='1':
        acq_mounted = check_mounted(acq_folder)
        if acq_mounted:
            status = sync_folder(names, acq_folder, local_backup)
            if status==0:
                move_folder_to_transferred(names, acq_folder)
        else:
            print('Warning!: ACQ not mounted, unable to check copy there')

    elif  res=='2':
        rack_mounted = check_mounted(server_folder)
        if rack_mounted:
            r_backup = server_folder +'/' +  subject_id
            if not path.exists(r_backup):
                mkdir(r_backup)
            rack_backup = server_folder +'/' +  subject_id + '/EMU'
            if not path.exists(rack_backup):
                mkdir(rack_backup)
            status = sync_folder(names, local_backup, rack_backup)
            if status==0:
                remove_folder_in_transferred(names, acq_folder, logfile)
        else:
            print('Warning!: RACK not mounted, unable to check copy there')

    if interactive:
        q=input('Enter or close to exit')
    exit()

