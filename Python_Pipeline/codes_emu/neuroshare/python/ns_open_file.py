from codecs import ignore_errors
import os
from tkinter import filedialog
import numpy as np
import sys

from collections import OrderedDict

def deblank(data):
    return data.rstrip('\x00 ')

def orderfields(s1, s2=None):
    """
    Order fields of a dictionary-like object.

    s1 : dict-like
        The dictionary-like object to be reordered.
    s2 : list or tuple, optional
        The order of fields to be used. If not provided, the fields will be
        sorted in ASCII dictionary order.

    Returns
    -------
    OrderedDict
        A new ordered dictionary with the fields in the specified order.

    Raises
    ------
    ValueError
        If s2 is provided and contains fields not present in s1.

    """
    if s2 is None:
        # Sort fields in ASCII dictionary order
        sorted_fields = sorted(s1.keys())
    else:
        # Use specified field order
        if set(s2) != set(s1.keys()):
            raise ValueError("s2 contains fields not present in s1")
        sorted_fields = s2

    # Create new ordered dictionary with fields in specified order
    return OrderedDict([(field, s1[field]) for field in sorted_fields])

def ns_OpenFile(*args):
    ns_RESULT = []
    hFile = {}
    
    if len(args) == 0 or (len(args) == 1 and args[0] == 'single'):
        path = filedialog.askopenfilename(title='Nev/NSX Loader')
        if path == '':
            return
        flag = len(args)
        pathname = os.path.dirname(path)
        name = os.path.basename(path)
        extT = os.path.splitext(name)[1]
        name = os.path.splitext(name)[0]
        
    elif (os.path.exists(args[0]) and args[1] == 'single'):
        path = args[0]
        pathname = os.path.dirname(path)
        name = os.path.basename(path)
        extT = os.path.splitext(name)[1]
        name = os.path.splitext(name)[0]
        flag = len(args) - 1
    else:
        ns_RESULT = 'ns_FILEERROR'
        return
    
    if flag:
        ext = extT
    else:
        ext = '.n'
    
    files = [f for f in os.listdir(pathname) if f.startswith(name) and os.path.splitext(f)[1].startswith(ext)]
    nFiles = len(files)
    fileNames = [f for f in files]
    fileSizes = [os.path.getsize(os.path.join(pathname, f)) for f in files]
    
    ns_RESULT = ['ns_OK' for i in range(nFiles)]
    hFile['Name'] = name
    hFile['FilePath'] = pathname
    hFile['TimeSpan'] = 0
    hFile['Entity'] = []
    hFile['FileInfo'] = [{} for i in range(nFiles)]
    
    for i in range(nFiles):
        fid = open(os.path.join(pathname, fileNames[i]), 'rb')
        # read the whole file into fcontents
        # raw_contents = fid.read()
        # # encoding = chardet.detect(raw_contents)['encoding']
        # fcontents = raw_contents.decode('utf-8', errors='ignore')
        # reset the file pointer to the beginning of the file
        # fid.seek(0, 1)
        type = fileNames[i][-3:] # file extension (nsx/nev)
        fileType = fid.read(8).decode('utf-8')
        hFile['FileInfo'][i]['FileID'] = fid
        hFile['FileInfo'][i]['Type'] = type
        hFile['FileInfo'][i]['FileSize'] = fileSizes[i]
        hFile['FileInfo'][i]['FileTypeID'] = fileType

        if fileType == 'NEURALEV': # if nev file
            # skip: File Spec and Additional Flags header Information 
            fid.seek(4, 1)
            hFile['FileInfo'][i]['Label'] = 'neural events'
            hFile['FileInfo'][i]['Period'] = 1
            hFile['FileInfo'][i]['BytesHeaders'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
            hFile['FileInfo'][i]['BytesDataPacket'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
            # skip: Time Resolution of Time Stamps, Time Resolution of Samples,
            # Time Origin, Application to Create File, and Comment field.
            fid.seek(312, 1)
            
            nExtendedHeaders = np.fromfile(fid, dtype=np.uint32, count=1)[0]
            # Get list of extended header PacketIDs: (there are 9 different nev
            # extended headers. Each have a 8 char array PacketID and each
            # extended header has 24 bytes of information with a total size of
            # 8+24=32 bytes. 
            PacketIDs = [np.fromfile(fid, dtype='S8', count=1)[0].decode('utf-8')]
            PacketIDs += [np.fromfile(fid, dtype='S8', count=1, offset=24)[0].decode('utf-8') for _ in range(nExtendedHeaders-1)]
            
            # Get Index of NEUEVWAV extended headers.
            idxEVWAV = np.where(PacketIDs == 'NEUEVWAV')[0]
            hFile['Entity'] += [{} for _ in range(len(idxEVWAV))]
            for j in range(len(idxEVWAV)):  
                # seek to the next NEUEVWAV extended header from the begining
                # of the file. 
                fid.seek(344+(idxEVWAV[j]-1)*32, 0)
                hFile['Entity'][j]['FileType'] = i + 1
                hFile['Entity'][j]['EntityType'] = 'Segment'     
                # Adding these variables to the structure here though they
                # are not part of the Segment entities so that the Entity
                # structure always has the same variables in it.
                hFile['Entity'][j]['Reason'] = 0
                hFile['Entity'][j]['Units'] = 'uV'
                hFile['Entity'][j]['Count'] = 0
                hFile['Entity'][j]['ElectrodeID'] = np.fromfile(fid, dtype=np.uint16, count=1)
                # skip: Physical Connector
                fid.seek(2, 1)
                # scale factor should convert bits to microvolts (nanovolts natively)
                hFile['Entity'][j]['Scale'] = np.fromfile(fid, dtype=np.uint16, count=1) * 10**-3
                # skip: Energy Threshold, High Threshold, Low Threshold
                fid.seek(6, 1)
                hFile['Entity'][j]['nUnits'] = np.fromfile(fid, dtype=np.uint8, count=1)
                # if scale factor = 0, use stim amp digitization factor
                if not hFile['Entity'][j]['Scale']:
                    # skip: Bytes per Waveform
                    fid.seek(1, 1)
                    # scale fact should convert bits to volts (volts natively)
                    hFile['Entity'][j]['Scale'] = np.fromfile(fid, dtype=np.float32, count=1)
                    hFile['Entity'][j]['Units'] = 'V'
                
            nDataPackets = (hFile['FileInfo'][i]['FileSize'] - float(hFile['FileInfo'][i]['BytesHeaders'])) / float(hFile['FileInfo'][i]['BytesDataPacket'])
              
            # Get Index of NEUEVLBL extended headers so we can put labels on the entities
            # NOTE: For NEV files saved with versions of Trellis earlier than 1.6, the NEUEVLBL
            # header is not present in the file
            idxEVLBL = np.where(PacketIDs == b'NEUEVLBL')[0]
            for j in range(len(idxEVLBL)):  
                # seek to the next NEUEVLBL extended header from the begining
                # of the file. 
                fid.seek(344+(idxEVLBL[j]-1)*32, -1)
                # get the electrode ID
                elecID = np.fromfile(fid, dtype=np.uint16, count=1)
                # find the entity with the matching electrode ID
                for k in range(len(hFile['Entity'])):
                    if hFile['Entity'][k]['ElectrodeID'] == elecID:
                        # get the label and store it on the entity
                        lbl = np.fromfile(fid, dtype='S16', count=1)
                        hFile['Entity'][j]['Label'] = lbl[0].rstrip('\x00 ')
                
            # Handle case where there is no spike or event data.  In this case,
            # we don't want to create any cache files, but we need to handle a
            # few variables.
            if nDataPackets == 0:
                hFile['FileInfo'][i]['MemoryMap'] = []
                if 'ElectrodeID' in hFile['Entity']:
                    hFile['FileInfo'][i]['ElectrodeList'] = [hFile['Entity']['ElectrodeID']]
                else:
                    hFile['FileInfo'][i]['ElectrodeList'] = []
                continue
            # create file to hold nev Event information. This file is memory
            # mapped for fast retrieval. 
            cacheFileName = os.path.join(pathname, name + '.cache')
            if not os.path.exists(cacheFileName):
                cacheID = open(cacheFileName, 'wb')
                fid.seek(hFile['FileInfo'][i]['BytesHeaders'], -1)
                # write timestamps to cache file
                np.fromfile(fid, dtype=np.uint32, count=nDataPackets).tofile(cacheID)
                fid.seek(hFile['FileInfo'][i]['BytesHeaders'] + 4, -1)
                # write Packet ID to cache file
                np.fromfile(fid, dtype=np.uint16, count=nDataPackets).tofile(cacheID)
                fid.seek(hFile['FileInfo'][i]['BytesHeaders'] + 6, -1)
                # write Classification/Insertion Reason to cache file
                np.fromfile(fid, dtype=np.uint8, count=nDataPackets).tofile(cacheID)
                cacheID.close()
            # create memory map of cache file
            hFile['FileInfo'][i]['MemoryMap'] = np.memmap(cacheFileName, dtype=[('TimeStamp', '<u4'), ('PacketID', '<u2'), ('Class', '<u1')], mode='r', shape=(nDataPackets, 1))
            # get a list of ElectrodesIDs that have a neural event
            uPacketID = np.unique(hFile['FileInfo'][i]['MemoryMap']['PacketID'])
            # get list of all ElectrodeIDs
            if 'ElectrodeID' in hFile['Entity']:
                allChan = hFile['Entity']['ElectrodeID']
                # Remove Entities that do not have neural events from the entity
                # list
                hFile['Entity'] = hFile['Entity'][np.isin(allChan, uPacketID[uPacketID!=0])]
            # get number of occurences of each ElectrodeID in nev file
            if len(hFile['Entity']) > 0:
                Count = [np.sum(hFile['FileInfo'][i]['MemoryMap']['PacketID'] == x) for x in hFile['Entity']['ElectrodeID']]
                hFile['Entity']['Count'] = Count
            # Calculate the Timespan in 30khz
            hFile['FileInfo'][i]['TimeSpan'] = float(hFile['FileInfo'][i]['MemoryMap']['TimeStamp'][-1])
            # update hFile TimeSpan if necessary
            if hFile['TimeSpan'] < hFile['FileInfo'][i]['TimeSpan']:
                hFile['TimeSpan'] = hFile['FileInfo'][i]['TimeSpan']
            # create entities for DIGITAL channels
            if not uPacketID[0]:
                # get all digital events
                eventClass = hFile['FileInfo'][i]['MemoryMap']['Class'][hFile['FileInfo'][i]['MemoryMap']['PacketID'] == 0]
                packetReason = ['Parallel Input', 'SMA 1', 'SMA 2', 'SMA 3', 'SMA 4', 'Output Echo']
                EC = len(hFile['Entity'])
                k = EC
                
                # Get Index of DIGLABEL extended headers so we can put labels 
                # on the entities NOTE: For NEV files saved with versions of 
                # Trellis earlier than 1.7, the fill will not contain multiple
                # DIGLABEL headers
                idxDIGLBL = np.where(PacketIDs == b'DIGLABEL')[0]
                digLbls = ['']*6
                if len(idxDIGLBL) == 5:
                    digLbls = []
                    for j in range(len(idxDIGLBL)):  
                        # seek to the next DIGLABEL extended header from the 
                        # begining of the file. 
                        fid.seek(344+(idxDIGLBL[j]-1)*32, -1)
                        # shift index because DIGLABEL headers for SMA come before Parallel
                        idx = (j%5) + 1 
                        # get the digital channel label 
                        lbl = np.fromfile(fid, dtype='S16', count=1)
                        digLbls.append(lbl[0].rstrip('\x00 '))
                        # get the digital mode
                        mode = np.fromfile(fid, dtype=np.uint8, count=1)
                        # get the electrode ID
                        elecID = np.fromfile(fid, dtype=np.uint16, count=1)
                    # there is no DIGLABEL header for output echo so we need to fake it
                    digLbls.append('Output Echo')
                
                # create Entities for digital channels that have events in the file
                for j in range(6):
                    Count = np.sum(np.bitwise_and(eventClass, 2**j))
                    if Count:
                        hFile['Entity'][k]['FileType'] = i + 1
                        hFile['Entity'][k]['EntityType'] = 'Event'
                        hFile['Entity'][k]['Reason'] = packetReason[j]
                        hFile['Entity'][k]['Label'] = digLbls[j]
                        hFile['Entity'][k]['Count'] = Count
                        hFile['Entity'][k]['ElectrodeID'] = np.uint16(0)
                        k=k+1
            hFile['FileInfo'][i]['ElectrodeList'] = [hFile['Entity']['ElectrodeID']]
            
            # Setup Neural Entities
            
            # Setup of 'Entity' struct held by the Neuroshare hFile
            # This needs to match the struct array in hFile['Entity']
            NeuralData = {'ElectrodeID': 0, 'EntityType': 'Neural', 'Reason': -1, 'Count': 0, 'Scale': 1, 'Units': '', 'nUnits': 0, 'FileType': i}
            # Get a list of all unique neural entities that have been found
            classList = np.sort(np.unique(hFile['FileInfo'][i]['MemoryMap']['Class']))
            # Find out how many unique electrodes and classes we have
            nElectrode = len(hFile['FileInfo'][i]['ElectrodeList'][hFile['FileInfo'][i]['ElectrodeList'] != 0])
            nClass = len(classList)
            # create space for the Neural entities.  This array will be put
            # at the end of the hFile['Entity'] after the Analog entities if 
            # count > 0
            neuralEntities = np.tile(NeuralData, (nElectrode*nClass, 1))
            # Create an entity for each possible electrode and class.  A given
            # electrode and class is mapped to it's electrode id and class by
            # index = electrode_index + class_index*nElectrodes
            for iEntity in range(nElectrode):
                elecID = hFile['FileInfo'][i]['ElectrodeList'][iEntity]
                if elecID == 0:
                    indices = hFile['FileInfo'][i]['MemoryMap']['PacketID'] == elecID
                    classes = hFile['FileInfo'][i]['MemoryMap']['Class'][indices]
                    for iClass in range(nClass):
                        class_ = classList[iClass]
                        neuralIndex = iEntity + iClass*nElectrode
                        neuralEntities[neuralIndex]['ElectrodeID'] = elecID
                        neuralEntities[neuralIndex]['Reason'] = class_
                        neuralEntities[neuralIndex]['Count'] = np.sum(classes==class_)
            # Remove all entities that were not represented in the data.  Now
            # We hold on to this array until AFTER all NSx files are read.
            hFile['Entity'] = np.concatenate((hFile['Entity'][np.where(hFile['Entity']['Count']>0)], neuralEntities[np.where(neuralEntities['Count']>0)]))

        elif fileType == 'NEURALSG': # nsx 2.1
            hFile['FileInfo'][i]['Label'] = np.fromfile(fid, dtype=np.uint8, count=16).tostring().decode('utf-8').rstrip('\x00 ')
            hFile['FileInfo'][i]['Period'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
            chanCount = np.fromfile(fid, dtype=np.uint32, count=1)[0]
            hFile['FileInfo'][i]['ElectrodeList'] = np.fromfile(fid, dtype=np.uint32, count=chanCount)
            hFile['FileInfo'][i]['BytesHeaders'] = fid.tell()
            EC = len(hFile['Entity'])
            # calculate the number of data points
            nDataPoints = (hFile['FileInfo'][i]['FileSize'] - hFile['FileInfo'][i]['BytesHeaders']) // (2 * chanCount)
            hFile['Entity'] += [{} for _ in range(chanCount)]
            for j in range(EC, EC+chanCount):
                hFile['Entity'][j]['FileType'] = i + 1
                hFile['Entity'][j]['EntityType'] = 'Analog'
                hFile['Entity'][j]['ElectrodeID'] = hFile['FileInfo'][i]['ElectrodeList'][j-EC]
                hFile['Entity'][j]['Scale'] = 1
                hFile['Entity'][j]['Units'] = 'V'
                hFile['Entity'][j]['Count'] = nDataPoints
            # store timeStamp and number of points in each packet
            hFile['FileInfo'][i]['TimeStamps'] = np.array([0, nDataPoints])
            # calculate time span
            hFile['FileInfo'][i]['TimeSpan'] = nDataPoints * hFile['FileInfo'][i]['Period']
            # update timespan if necessary
            if hFile['TimeSpan'] < hFile['FileInfo'][i]['TimeSpan']:
                hFile['TimeSpan'] = hFile['FileInfo'][i]['TimeSpan']
        elif fileType == 'NEURALCD' or fileType == 'NEUCDFLT':
            # skip: File Spec
            if fileType == 'NEUCDFLT':
                floatStream = 1
            else:
                floatStream = 0

            fid.seek(2, 1)
            hFile['FileInfo'][i]['BytesHeaders'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
            hFile['FileInfo'][i]['Label'] = fid.read(16).decode('utf-8').rstrip('\x00 ')
            # skip: Comment
            fid.seek(256, 1)
            hFile['FileInfo'][i]['Period'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
            # skip: Time Resolution of Time Stamps and Time Origin
            fid.seek(20, 1)
            chanCount = np.fromfile(fid, dtype=np.uint32, count=1)[0]
            EC = len(hFile['Entity'])
            hFile['Entity'] += [{} for _ in range(chanCount)]
            # get information from extended headers
            for j in range(EC, EC+chanCount):
                hFile['Entity'][j]['FileType'] = i + 1
                hFile['Entity'][j]['EntityType'] = 'Analog'
                # skip: Type
                fid.seek(2, 1)
                hFile['Entity'][j]['ElectrodeID'] = np.fromfile(fid, dtype=np.uint16, count=1)[0]
                lbl = np.fromfile(fid, dtype=np.uint8, count=16).tostring().decode('utf-8').rstrip('\x00 ')
                hFile['Entity'][j]['Label'] = lbl
                # skip: Physical connector, Connector pin
                fid.seek(2, 1)
                # read Min/Max Digital and Min/Max Analog values
                analogScale = np.fromfile(fid, dtype=np.int16, count=4)
                if floatStream:
                    hFile['Entity'][j]['Scale'] = 1.0
                else:
                    hFile['Entity'][j]['Scale'] = (int(analogScale[3]) - int(analogScale[2])) / (int(analogScale[1]) - int(analogScale[0]))
                hFile['Entity'][j]['Units'] = np.fromfile(fid, dtype=np.uint8, count=16).tostring().decode('utf-8').rstrip('\x00 ')
                # skip: High/Low Freq Corner, High/Low Freq Order, Hight/Low Filter Type
                fid.seek(20, 1)
            hFile['FileInfo'][i]['ElectrodeList'] = [hFile['Entity'][j]['ElectrodeID'] for j in range(EC, EC+chanCount)]
            fid.seek(hFile['FileInfo'][i]['BytesHeaders'], 0)
            hFile['FileInfo'][i]['TimeStamps'] = np.empty((0, 2))
            while fid.tell() < hFile['FileInfo'][i]['FileSize']:
                # skip Data Packet header
                tell = fid.tell()
                header = np.fromfile(fid, dtype=np.uint8, count=1)[0]
                # read and store timeStamps and number of points in each packet
                timeStamps = np.fromfile(fid, dtype=np.uint32, count=1)[0] / hFile['FileInfo'][i]['Period']
                nPoints = np.fromfile(fid, dtype=np.uint32, count=1)[0]
                hFile['FileInfo'][i]['TimeStamps'] = np.vstack((hFile['FileInfo'][i]['TimeStamps'], [timeStamps, nPoints]))
                # skip to the next collection of data points
                if floatStream:
                    bytesPerPoint = 4
                else:
                    bytesPerPoint = 2
                file_sz = bytesPerPoint * int(nPoints) * int(chanCount) # fix for overflow
                rc = fid.seek(file_sz, 1)
                if not rc :
                    print('Warning: corrupted nsx file: {}.{}'.format(hFile['Name'], hFile['FileInfo'][i]['Type']), file=sys.stderr)
                    break
            # get number of points of each Entity
            for j in range(EC, EC+chanCount):
                hFile['Entity'][j]['Count'] = int(sum(hFile['FileInfo'][i]['TimeStamps'][:, 1]))
            # calculate time span
            hFile['FileInfo'][i]['TimeSpan'] = (sum(hFile['FileInfo'][i]['TimeStamps'][:, -1])) * hFile['FileInfo'][i]['Period']
            # update timespan if necessary
            if hFile['TimeSpan'] < hFile['FileInfo'][i]['TimeSpan']:
                hFile['TimeSpan'] = hFile['FileInfo'][i]['TimeSpan']
        else: # if no valid File Type ID, output fileError 
            ns_RESULT[i] = 'ns_FILEERROR'
            continue
        hFile = orderfields(hFile)
    hFile['Entity'] = [entity for entity in hFile['Entity'] if entity['Count'] > 0]

    if 'neuralEntities' in locals() and neuralEntities:
        hFile['Entity'] += neuralEntities

    if 'ns_OK' in ns_RESULT:
        ns_RESULT = 'ns_OK'
    else:
        hFile = {}
        ns_RESULT = 'ns_FILEERROR'


    return ns_RESULT, hFile


