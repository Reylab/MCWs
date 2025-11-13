import os
import numpy as np
import re
from qtpy.QtCore import Qt
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "neuroshare/pyns"))
)
sys.path.append(os.path.dirname(__file__))
from nsentity import EntityType
# Map file sample:

class RippleElectrodeInfo:
    """
    Class to store information about a single electrode.
    """

    def __init__(self, port=None, fe=None, ch_id=None, id=None, label=None, bundle=None, enabled=False):
        self.port = port # A, B, C, D, AIO, DIO
        self.fe = fe # 1-4
        self.ch_id = ch_id # 1-32
        self.id = id # 0-10245
        self.label = label
        self.bundle = bundle
        self.enabled = enabled
    
    def __eq__(self, other):
        if isinstance(other, RippleElectrodeInfo):
            return self.id == other.id
        else:
            return False
        
    def __hash__(self):
        return hash(self.id)

class RippleMapFile:
    """
    Class to read and write ripple map files.
    """

    def __init__(self, filename=None):
        self.electrodes = []
        self.bundles = []
        self.filename = filename

        self.init_map_file()

        if filename is not None:
            self.read_ripple_map(filename)

    def init_map_file(self):
        """
        Initialize a ripple map file. Summit has four ports A, B, C, and D.
        Each port can have 4 front-ends with 32 channels each.
        Other inputs are from explorer analog+digital IOs, audio.
        """
        self.port_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'AIO': 4, 'DIO': 5}
        self.fe_dict = {'1': 0, '2': 1, '3': 2, '4': 3}
        id = 0
        for p_idx, port in enumerate(['A', 'B', 'C', 'D']):
            for fe in range(4):
                for ch in range(32):
                    ch_id = ch + 1
                    label = f'{port}{fe}{ch}'
                    self.electrodes.append(RippleElectrodeInfo(port=port, fe=fe, ch_id=ch_id, id=id, label=label))
                    id += 1

        # Add explorer inputs
        # 1.DIO.BNC.001; Photo_digital; ;
        # 1.DIO.PAR.001; parallel_dig; ;
        # 1.AIO.AUD.001; MicL; ;
        # 1.AIO.AUD.002; MicR; ;
        # 1.AIO.BNC.001; Photo_analog; ;
        self.photo_analog = RippleElectrodeInfo(port='AIO', fe='BNC', ch_id=1, id=10240, label='Photo_analog', enabled=True, bundle='analog')
        self.electrodes.append(self.photo_analog)
        self.electrodes.append(RippleElectrodeInfo(port='DIO', fe='BNC', ch_id=1, id=10241, label='Photo_digital', enabled=True, bundle='digital'))
        self.electrodes.append(RippleElectrodeInfo(port='DIO', fe='PAR', ch_id=1, id=10242, label='parallel_dig', enabled=True, bundle='digital'))
        self.electrodes.append(RippleElectrodeInfo(port='AIO', fe='BNC', ch_id=1, id=10243, label='Mic2', enabled=True, bundle='audio'))
        self.electrodes.append(RippleElectrodeInfo(port='AIO', fe='AUD', ch_id=1, id=10244, label='MicL', enabled=True, bundle='audio'))
        self.electrodes.append(RippleElectrodeInfo(port='AIO', fe='AUD', ch_id=2, id=10245, label='MicR', enabled=True, bundle='audio'))

        self.init_channel_dict()

    def init_channel_dict(self):
        """ 
        Initialize the channel dictionary.
        """
        self.channel_dict = {}
        for port in self.port_dict:
            self.channel_dict[port] = (False, {})

    def update_channel_dict(self, channel_dict):
        """
        Update the channel dictionary.
        """
        for port, bundles in channel_dict.items():
            for bundle, electrodes in bundles[1].items():
                for label, electrode in electrodes[1].items():
                    elec = self.channel_dict[port][1][bundle][1][label]
                    if electrode[0] == Qt.Checked or electrode[0] == Qt.PartiallyChecked:
                        self.channel_dict[port][1][bundle][1][label] = (True, elec[1])
                    else:
                        self.channel_dict[port][1][bundle][1][label] = (False, elec[1])

    def create_ripple_map_nsx(self, entities):
        """
        Create ripple map from nsx entities.
        """
        self.init_channel_dict()
        self.electrodes = []
        self.bundles = []
        for entity in entities:
            if entity.entity_type == EntityType.event:
                continue
            id = entity.electrode_id - 1
            if id < 128:
                port = 'A'
            elif id < 256:
                port = 'B'
            elif id < 384:
                port = 'C'
            elif id < 512:
                port = 'D'
            elif id == 10240:
                port = 'AIO'
            elif id == 10241:
                port = 'DIO'
            else:
                continue
            elec_info = RippleElectrodeInfo(port=port, fe=entity.electrode_id//4, 
                                            ch_id=id, id=entity.electrode_id, 
                                            label=entity.electrode_label)
            
            elec_info.bundle = re.sub(r'\d+|ref|raw', '', entity.electrode_label)
            if elec_info.bundle is not None and elec_info.bundle not in self.bundles:
                self.bundles.append(elec_info.bundle)
                self.channel_dict[elec_info.port][1][elec_info.bundle] = (True, {})
            self.electrodes.append(elec_info)
        for electrode in self.electrodes:
            self.channel_dict[electrode.port][1][electrode.bundle][1][electrode.label] = (True, electrode)
                    
    def read_ripple_map(self, filename=None):
        """
        Read a ripple map file.

        Parameters
        ----------
        filename : str
            The filename of the ripple map file to read. If not provided, the
            filename provided to the constructor will be used.
        """
        if filename is None:
            filename = self.filename

        file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../', 'assets', 'map_files', filename))

        if not os.path.exists(file_path):
            print(f'File {file_path} does not exist.')
            return
        
        self.bundles = []
        # read all the lines
        with open(file_path, 'rb') as f:
            for line in f:
                # Check if line starts with # and ignore it
                if line.startswith(b'#') or line.startswith(b'\n'):
                    continue
                else:
                    line_info = line.decode('utf-8').strip()
                    if line_info == '':
                        continue
                    line_info = line_info.split(';')
                    port = line_info[0].split('.')[1]
                    # 1.C.4.001; mLAMY01; ;
                    # 1.C.4.002; mLAMY02; ;

                    # 1.DIO.BNC.001; Photo_digital; ;
                    # 1.DIO.PAR.001; parallel_dig; ;
                    # 1.AIO.AUD.001; MicL; ;
                    # 1.AIO.AUD.002; MicR; ;
                    # 1.AIO.BNC.001; Photo_analog; ;
                    # 1.AIO.BNC.002; Mic2; ;
                    if port == 'AIO' or port == 'DIO':
                        label = line_info[1].split()[-1]
                        if port == 'AIO':
                            if 'analog' not in self.bundles and label.lower() == 'photo_analog':
                                self.bundles.append('analog')
                                self.channel_dict[port][1]['analog'] = (True, {})
                            if 'audio' not in self.bundles and label.lower().find('mic') != -1:
                                self.bundles.append('audio')
                                self.channel_dict[port][1]['audio'] = (True, {})
                        else:
                            if 'digital' not in self.bundles and label.lower().find('dig') != -1:
                                self.bundles.append('digital')
                                self.channel_dict[port][1]['digital'] = (True, {})
                    else:
                        fe = line_info[0].split('.')[2]
                        id = int(line_info[0].split('.')[-1]) - 1
                        id = self.port_dict[port] * 128 + self.fe_dict[fe] * 32 + id
                        label = line_info[1].split()[-1]
                        self.electrodes[id].label = label
                        self.electrodes[id].enabled = True
                        self.electrodes[id].bundle = re.sub(r'\d+|ref', '', label)

                        if self.electrodes[id].bundle is not None and self.electrodes[id].bundle not in self.bundles:
                            self.bundles.append(self.electrodes[id].bundle)
                            self.channel_dict[self.electrodes[id].port][1][self.electrodes[id].bundle] = (True, {})

        for electrode in self.electrodes:
            if electrode.bundle is not None:
                self.channel_dict[electrode.port][1][electrode.bundle][1][electrode.label] = (True, electrode)
        


        print(f'Bundles: {self.bundles}')

    def get_port(self, port, fe=None):
        """
        Get all electrodes on a port.

        Parameters
        ----------
        port : str
            The port to get electrodes from.
        fe : int
            The front-end to get electrodes from. If not provided, all
            electrodes on the port will be returned.

        Returns
        -------
        list
            A list of electrodes on the port.
        """
        if fe is None:
            return [e for e in self.electrodes if e.port == port]
        else:
            return [e for e in self.electrodes if e.port == port and e.fe == fe]
        
    def get_bundle(self, bundle):
        """
        Get all electrodes in a bundle.

        Parameters
        ----------
        bundle : str
            The bundle to get electrodes from.

        Returns
        -------
        list
            A list of electrodes in the bundle.
        """
        return [e for e in self.electrodes if e.bundle == bundle]
    
    def get_micro_bundles(self):
        """
        Get all micro bundles.

        Returns
        -------
        list
            A list of micro bundles.
        """
        bun_dict = {}
        for bun in self.bundles:
            if bun.startswith('m'):
                bun_dict[bun] = self.get_bundle(bun)

        return bun_dict


        
