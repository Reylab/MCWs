import os
# import mat73
import scipy
import numpy as np
import pandas as pd
import time
import traceback

class StudyInfo():

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(StudyInfo, cls).__new__(cls)
        return cls.instance

    def __init__(self, c, nsx_file_path=None):
        self.c = c
        self.image_names = []
        if nsx_file_path is not None:
            self.study_folder = os.path.dirname(nsx_file_path)
            self.load_study_info()
        
        
    #region Load study info (sequence, stimuli, responses)
        
    def load_study_info(self, study_folder=None):
        """
        Loads the study info.
        """
        if study_folder is None:
            study_folder = self.study_folder

        pre_processing_info = os.path.join(study_folder, 'pre_processing_info.mat')
        if os.path.exists(pre_processing_info):
            self.notches_info = scipy.io.loadmat(pre_processing_info, simplify_cells=True)['process_info']

        pics_onset_file = os.path.join(study_folder, 'finalevents.mat')
        if os.path.exists(pics_onset_file):
            self.pics_onset = scipy.io.loadmat(os.path.join(study_folder, 'finalevents.mat'), simplify_cells=True)['pics_onset']
            self.pics_onset_1d = []
            for subscr in self.pics_onset:
                for seq in subscr.T:
                    self.pics_onset_1d.extend(seq.tolist())
            # convert to samples from ms
            self.pics_onset_1d = np.array(self.pics_onset_1d) * 30
            print(f'Pics onset: {len(self.pics_onset_1d)}')
        else:
            self.c.log.emit('finalevents.mat not found.')
        # stimulus_file = os.path.join(study_folder, 'stimulus.mat')
        # if os.path.exists(stimulus_file):
        #     self.stimulus = scipy.io.loadmat(os.path.join(study_folder, 'stimulus.mat'), simplify_cells=True)['stimulus']
        # else:
        #     self.c.log.emit('stimulus.mat not found.')
        
        pics_order_file = os.path.join(study_folder, 'experiment_properties_online3.mat')
        if os.path.exists(pics_order_file):
            self.exp_properties = scipy.io.loadmat(os.path.join(study_folder, 'experiment_properties_online3.mat'), simplify_cells=True)
            self.pics_order = scipy.io.loadmat(os.path.join(study_folder, 'experiment_properties_online3.mat'), simplify_cells=True)['scr_config_cell']
            self.set_pics_order_1d()
        else:
            self.c.log.emit('experiment_properties_online3.mat not found.')

        # grapes_file = os.path.join(study_folder, 'grapes.mat')
        # if os.path.exists(grapes_file):
        #     # self.grapes = scipy.io.loadmat(os.path.join(study_folder, 'grapes.mat'), simplify_cells=True)
        #     self.grapes = mat73.loadmat(grapes_file)
        #     # self.image_names = self.grapes['ImageNames'].tolist()
        #     print(self.grapes.keys())
        # else:
        #     self.c.log.emit('grapes.mat not found.')

        nsx_file = os.path.join(self.study_folder, 'NSx.mat')
        if os.path.exists(nsx_file):
            self.nsx = scipy.io.loadmat(nsx_file, simplify_cells=True)   
        else:
            self.c.log.emit('Nsx.mat not found')

        ifr_file = os.path.join(self.study_folder, 'ifr.mat')
        if os.path.exists(ifr_file):
            ifrmat_start = time.time()
            self.ifr = scipy.io.loadmat(ifr_file, simplify_cells=True)
            # self.ifr = h5py.File(ifr_file, 'r')
            
            # self.ifr = mat73.loadmat(ifr_file)
            print(f'IFR loaded in {time.time()-ifrmat_start:.2f} seconds.')

        ranking_table_start = time.time()
        ranking_table_file = os.path.join(self.study_folder, 'sorted_table.csv')
        if os.path.exists(ranking_table_file):
            self.read_ranking_table(ranking_table_file=ranking_table_file)
            # self.display_sorted_table()
        else:
            self.c.log.emit("Ranking table not found (sorted_table.csv)")

        print(f'Ranking table loaded in {time.time()-ranking_table_start:.2f} seconds.')

        stimuli_table_file = os.path.join(self.study_folder, 'image_info_table.csv')
        if os.path.exists(stimuli_table_file):
            self.stimuli_table = pd.read_csv(stimuli_table_file, header=0)
            self.image_names = self.stimuli_table['name'].tolist()
            print(self.stimuli_table.head(5))
        else:
            self.c.log.emit(f'{stimuli_table_file} not found')

        rasters_start = time.time()
        rasters_file = os.path.join(self.study_folder, 'rasters.mat')
        if os.path.exists(rasters_file):
            self.rasters = scipy.io.loadmat(rasters_file, simplify_cells=True)
        else:
            print(f'{rasters_file} not found.')

        print(f'Rasters loaded in {time.time()-rasters_start:.2f} seconds.')

        print('Study info loaded.')

    def read_ranking_table(self, ranking_table_file):
        ranking_table_read_start = time.time()
        self.ranking_table = pd.read_csv(ranking_table_file)
        print(self.ranking_table.head(5))
        print(f'Ranking table read in {time.time()-ranking_table_read_start} seconds.')

        # Change channel ids to channel names
        # Get unique channels from the ranking table
        channels_ids = self.ranking_table['channel'].unique()

        # Get channel names from the nsx file
        self.channels = []
        self.bundles = {}
        for ch_id in channels_ids:
            elec_id = ch_id[4:]
            
            for ch in self.nsx['NSx']:
                if ch['chan_ID'] == int(elec_id):
                    self.channels.append(ch['output_name'])
                    try:
                        if ch['bundle'] not in self.bundles:
                            self.bundles[ch['bundle']] = []
                        self.bundles[ch['bundle']].append(ch['output_name'])
                    except:
                        print(traceback.format_exc())
                    break

    def set_pics_order_1d(self):
        """
        Gets the order of the stimulus images from the pics_order variable.
        """
        self.pics_order_1d = []
        for block in self.pics_order:
            for seq in block['order_pic'].T:
                for pic in seq:
                    self.pics_order_1d.append(block['pics2use'][pic-1])

        self.used_pics, self.used_pics_count = np.unique(self.pics_order_1d, return_counts=True)
    #endregion Load study info (sequence, stimuli, responses)