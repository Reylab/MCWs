import numpy as np
import os
from scipy.io import loadmat, savemat

class WithinChannelArtifact:
    """
    Implements within-channel artifact rejection for extreme amplitude spikes.
    Works on output from Collision.separate_collisions().
    """

    def __init__(self, channels):
        self.channels = channels
        self.amp_factor = 10  # factor to multiply spike detection threshold

    def extreme_amplitude_detection(self):
        """
        For each channel, quarantines spikes exceeding amp_factor * detection threshold.
        Updates the .mat files by flagging or removing spikes.
        """
        directory = os.path.dirname(os.path.dirname(__file__))

        for ch in self.channels:
            mat_files = [f for f in os.listdir(directory+'/output/') if f.endswith(f"{ch}_spikes.mat")]
            if not mat_files:
                print(f"No spike file found for channel {ch}")
                continue

            mat_file = directory + '/output/' + mat_files[0]
            data = loadmat(mat_file)

            spikes = data['spikes']
            threshold = data['threshold'][0]  # threshold per segment
            possible_artifact = data.get('possible_artifact', np.zeros(spikes.shape[0], dtype=bool))

            # Determine threshold for extreme amplitude
            thr_extreme = np.array(threshold) * self.amp_factor

            # Quarantine spikes exceeding extreme amplitude
            extreme_idx = np.any(np.abs(spikes) > thr_extreme[:, np.newaxis], axis=1)
            possible_artifact = np.logical_or(possible_artifact, extreme_idx)

            # Optionally remove extreme spikes now or keep quarantined for rescue
            spikes_clean = spikes[~possible_artifact, :]

            # Save updated mat file
            mdict = {
                'par': data['par'],
                'threshold': threshold,
                'index_all': data['index_all'],
                'index': data['index'][0, ~possible_artifact],  # only keep non-extreme
                'spikes': spikes_clean,
                'spikes_all': data['spikes_all'],
                'possible_artifact': possible_artifact,
                'psegment': data['psegment'],
                'sr_psegment': data['sr_psegment']
            }
            savemat(mat_file, mdict)
            print(f"Channel {ch}: Extreme amplitude artifacts quarantined: {np.sum(extreme_idx)}/{spikes.shape[0]}")
