import numpy as np
from scipy.io import loadmat, savemat
import os
import concurrent.futures
import multiprocessing
import math
import glob


class Collision():
    def __init__(self, channels, nsx):
        self.channels = channels
        self.nsx = nsx

    def separate_collisions(self, channels='all'):
        """
        Separate collisions for specified channels.
        Creates mask_bundle_coll field in spike files to mark collision spikes.
        
        Parameters:
        -----------
        channels : 'all', list, int, optional
            - 'all': Process all channels (default)
            - list: Specific channel numbers [257, 263, 290]
            - int: Single channel number 257
            
        Examples:
        ---------
        col.separate_collisions(channels='all')
        col.separate_collisions(channels=[257, 263])
        col.separate_collisions(channels=257)
        col.separate_collisions(channels=list(range(290, 301)))
        """
        # ========== CHANNEL FILTERING ==========
        # Get all available channels from NSx
        all_channels_raw = self.nsx['chan_ID'][0]
        
        if isinstance(channels, str) and channels.lower() == 'all':
            # Use all channels - keep original format
            selected_channels = all_channels_raw
            selected_nums = [ch[0] for ch in all_channels_raw]
            print(f"Processing collisions for all {len(selected_channels)} channels: {selected_nums}")
        else:
            # Convert to list if single channel
            if isinstance(channels, int):
                channels = [channels]
            
            # Filter channels - KEEP ORIGINAL CHANNEL OBJECTS
            selected_channels = np.array([ch for ch in all_channels_raw if ch[0] in channels], dtype=object)
            
            print(f"Requested channels: {channels}")
            print(f"Found {len(selected_channels)} matching channels: {[ch[0] for ch in selected_channels]}")
            
            # If no matches, warn and use all
            if len(selected_channels) == 0:
                print(f"WARNING: No matching channels found!")
                print(f"Available channels: {[ch[0] for ch in all_channels_raw]}")
                print("Using all channels instead.")
                selected_channels = all_channels_raw
        # ========================================
        
        # Unpack nsx with selected channels
        nsx_idxs = np.isin(self.nsx['chan_ID'], selected_channels)[0]
        self.t_win = 0.5
        self.bundle_min_art = 6
        bundles = self.nsx['bundle'].ravel()[nsx_idxs]
        is_micros = self.nsx['is_micro'].ravel()[nsx_idxs]
        output_names = self.nsx['output_name'].ravel()[nsx_idxs]
        chan_IDs = self.nsx['chan_ID'].ravel()[nsx_idxs]
        bundles_to_explore = np.unique(bundles)
        
        directory = os.path.dirname(os.path.dirname(__file__))
        
        for ibun, bundle_to_explore in enumerate(bundles_to_explore):
            pos_chans_probe = np.nonzero(bundles == bundle_to_explore)[0]
            if not is_micros[ibun]:
                continue
            
            if 'Mic' in bundle_to_explore[0] or 'Photo' in bundle_to_explore[0]:
                continue
            
            all_spktimes = np.array([])
            which_chan = np.array([])
            spktimes = np.array([])
            
            # ========== LOAD SPIKES FROM SELECTED CHANNELS ==========
            for k, pos_chan_probe in enumerate(pos_chans_probe):
                spk = loadmat(directory + '/output/%s_spikes.mat' % output_names[pos_chan_probe][0], 
                             mdict={'t_win': self.t_win, 'bundle_min_art': self.bundle_min_art})
                
                if 'index_all' in spk.keys():
                    spktimes = np.array(spk['index_all'])[0]
                    if not all_spktimes.size:
                        all_spktimes = spktimes
                    else:
                        all_spktimes = np.append(all_spktimes, spktimes)
                else:
                    spktimes = np.array(spk['index'])[0]
                    if not all_spktimes.size:
                        all_spktimes = spktimes 
                    else:
                        all_spktimes = np.append(all_spktimes, spktimes)
                
                if not which_chan.size:
                    which_chan = chan_IDs[pos_chans_probe[k]][0][0] * np.ones(spktimes.shape, dtype=np.uint16)
                else:
                    which_chan = np.append(which_chan, chan_IDs[pos_chans_probe[k]][0][0] * np.ones(spktimes.shape, dtype=np.uint16))
            
            # ========== SORT SPIKES BY TIME ==========
            ii = np.argsort(all_spktimes)
            all_spktimes = np.sort(all_spktimes)
            which_chan = which_chan[ii]
            is_artifact = np.zeros(np.shape(all_spktimes))
            artifact_idxs = np.array([])

            # ========== DETECT ARTIFACTS ==========
            # TODO: add multiprocessing for this part below
            for ispk in range(all_spktimes.size):
                which_spks = np.nonzero((all_spktimes >= all_spktimes[ispk]) & (all_spktimes < all_spktimes[ispk] + self.t_win))[0]
                if np.size(np.unique(which_chan[which_spks])) >= self.bundle_min_art:
                    artifact_idxs = np.append(artifact_idxs, which_spks)

            artifact_idxs = np.unique(artifact_idxs.astype(np.int64))
            is_artifact[artifact_idxs] = True
            
            # ========== SAVE FILTERED SPIKES WITH MASK ==========
            for k, pos in enumerate(pos_chans_probe):
                spk = loadmat(directory + '/output/' + '%s_spikes.mat' % output_names[pos][0])
                
                if 'index_all' in spk.keys():
                    spikes_all = spk['spikes_all']
                    index_all = spk['index_all']
                else:
                    spikes_all = spk['spikes']
                    index_all = spk['index']
                
                par = spk['par']
                threshold = spk['threshold']
                psegment = spk['psegment']
                sr_psegment = spk['sr_psegment']
                possible_artifact = spk['possible_artifact']

                # ===== CREATE COLLISION MASK =====
                # mask_bundle_coll: 1 for collisions, 0 for non-collisions
                # Size: same as spikes_all
                mask_bundle_coll = np.ones(spikes_all.shape[0], dtype=np.uint8)
                
                # Get non-collision spikes for this channel
                index = all_spktimes[np.invert(is_artifact.astype(np.bool_)) & (which_chan == chan_IDs[pos][0][0])]
                
                # Mark non-collision spikes as 0
                collision_mask_idxs = np.where(np.isin(index_all.ravel(), index))[0]
                mask_bundle_coll[collision_mask_idxs] = 0
                
                spikes = spikes_all[np.isin(index_all, index).ravel(), :]
                
                # Setup mat structure with dict
                mdict = {
                    'par': par,
                    'threshold': threshold,
                    'index_all': index_all,
                    'index': index,
                    'spikes': spikes,
                    'spikes_all': spikes_all,
                    'possible_artifact': possible_artifact,
                    'psegment': psegment,
                    'sr_psegment': sr_psegment,
                    'mask_bundle_coll': mask_bundle_coll
                }

                savemat(directory + '/output/' + '%s_spikes.mat' % output_names[pos_chans_probe[k]][0], mdict=mdict)
            
            print("Collision detection (%s). Total artifacts:%d/%d(%2.2f%%) \n" % 
                  (bundle_to_explore, artifact_idxs.size, all_spktimes.size, artifact_idxs.size / all_spktimes.size * 100))


if __name__ == '__main__':
    NSx_file_path = os.path.abspath(glob.glob("input/NSx.mat")[0])
    metadata = loadmat(NSx_file_path)
    nsx = metadata['NSx']
    
    # ========================================
    # CHANNEL SELECTION (NON-INTERACTIVE)
    # ========================================
    
    # Option 1: Use all channels
    specific_channels = 'all'
    
    # Option 2: Use specific channels
    # specific_channels = [257, 263]
    
    # Option 3: Use single channel
    # specific_channels = 259
    
    # Option 4: Use range of channels
    # specific_channels = list(range(290, 301))  # 290-300
    
    col = Collision(nsx['chan_ID'][0], nsx)
    col.separate_collisions(channels=specific_channels)