import os
import numpy as np
import datetime
import glob
import pandas as pd
import time
import sys
import concurrent.futures
import threading
file_dir = os.path.dirname(__file__)
sys.path.append(
    os.path.abspath(os.path.join(file_dir, "../..", "neuroshare/pyns"))
)
sys.path.append(file_dir)
sys.path.append(os.path.join(file_dir, ".."))

from nsfile import NSFile
from nsentity import EntityType
from core.utils import get_elapsed_time, get_time_hrs_mins_secs_ms

lock = threading.Lock()

def get_rec_start_time(filename):
    """
    Get the recording start time from the filename
    because pyns does not reliably get the correct time.
    """
    try:
        Date = None
        # Get recording start time
        with open(filename, "rb") as fid:
            fid.seek(294, os.SEEK_SET)
            Date = np.fromfile(fid, dtype=np.uint16, count=8)

        if Date is None or Date.size == 0:
            return filename, None, None, None
        time_year = Date[0]
        time_month = Date[1]
        time_day = Date[3]
        time_hour = Date[4]
        time_min = Date[5]
        time_sec = Date[6]
        time_millisec = Date[7]

        # Ripple records in UTC
        rec_start_time = datetime.datetime(
            time_year,
            time_month,
            time_day,
            time_hour,
            time_min,
            time_sec,
            time_millisec * 1000,
            datetime.timezone.utc,
        )

        # Convert rec_start_time from UTC to local time
        rec_start_time = rec_start_time.astimezone()

        ns_file = NSFile(filename, proc_single=True)
        rec_length = ns_file.get_file_info().time_span # in seconds
        if rec_length is None or rec_length < 5:
            print(f'Warning: {filename} has a recording length of {rec_length} seconds. ')
        rec_end_time = rec_start_time + datetime.timedelta(seconds=rec_length)

        # Convert rec_length to hours, minutes, seconds
        # rec_length = str(datetime.timedelta(seconds=rec_length))
        hrs, mins, secs, ms = get_time_hrs_mins_secs_ms(seconds=rec_length)
        rec_length = f'{hrs:.0f}:{mins:.0f}:{secs:.0f}'
    except:
        print(f'Error getting recording start time for {filename}')
        return filename, None, None, None

    return filename, rec_start_time, rec_end_time, rec_length

if __name__ == "__main__":
    folder = 'Z:\\'
    study_folders = []
    multi_thread = True

    df_nf3 = pd.DataFrame(columns=['nf3_file', 'recording_start_time', 'recording_end_time', 'duration'])

    # Get all the study folders in the folder
    study_folders = [f.path for f in os.scandir(folder) if f.is_dir() and 'MCW-FH' in f.name]
    nf3_files_list = []
    find_start_time = time.time()
    # Get all the nf3 files in the study folders
    for study_folder in study_folders:
        nf3_files = glob.glob(os.path.join(study_folder, 'EMU', '**', '*.nf3'), recursive=True)
        nf3_files_list.extend(nf3_files)
    print(f'Found {len(nf3_files_list)} nf3 files in {folder} in {time.time() - find_start_time} seconds')

    rec_start_time_start = time.time()
    total = len(nf3_files_list)
    if multi_thread:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(get_rec_start_time, nf3_file) for nf3_file in nf3_files_list]
            completed_count = 0
            
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                filename, rec_start_time, rec_end_time, rec_length = future.result()
                df_nf3.loc[i] = [filename, rec_start_time, rec_end_time, rec_length]

                completed_count += 1

                if completed_count >1349:
                    hrs, mins, secs, ms = get_elapsed_time(start_time=rec_start_time_start)
                    # Remaining time = elapsed time / completed tasks * remaining tasks
                    remaining_time = (hrs * 3600 + mins * 60 + secs) / completed_count * (total - completed_count)
                    hrs, mins, secs, ms = get_time_hrs_mins_secs_ms(seconds=remaining_time)
                    print(f'{completed_count}/{total} done. Estimated remaining time: {hrs:.0f} hours, {mins:.0f} minutes, {secs:.0f} seconds, {ms:.0f} ms.')
    else:
        for i, nf3_file in enumerate(nf3_files_list):
            filename, rec_start_time, rec_end_time, rec_length = get_rec_start_time(nf3_file)
            df_nf3.loc[i] = [filename, rec_start_time, rec_end_time, rec_length]
            completed_count = i + 1
            if completed_count % 100 == 0:
                hrs, mins, secs, ms = get_elapsed_time(start_time=rec_start_time_start)
                # Remaining time = elapsed time / completed tasks * remaining tasks
                remaining_time = (hrs * 3600 + mins * 60 + secs) / i * (total - completed_count)
                hrs, mins, secs, ms = get_time_hrs_mins_secs_ms(seconds=remaining_time)
                print(f'{completed_count}/{total} done. Estimated remaining time: {hrs:.0f} hours, {mins:.0f} minutes, {secs:.0f} seconds, {ms:.0f} ms.')

    # Sort the dataframe by recording start time
    df_nf3 = df_nf3.sort_values(by='recording_start_time')

    print(f'Got recording start times for {len(nf3_files_list)} nf3 files in {time.time() - rec_start_time_start} seconds')
    to_csv_start = time.time()
    save_path = os.path.join(file_dir, 'nf3_info.csv')
    df_nf3.to_csv(save_path, index=False)
    print(f'Saved recording start times to {save_path} in {time.time() - to_csv_start} seconds')