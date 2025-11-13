import numpy as np

def ns_GetSegmentSourceInfo(hFile, EntityID, SourceID=1):
    # Initialize output variables
    ns_RESULT = ''
    nsSegmentSourceInfo = {}

    # Check hFile
    if not isinstance(hFile, dict):
        ns_RESULT = 'ns_BADFILE'
        return ns_RESULT, nsSegmentSourceInfo

    # Check Entity
    if not isinstance(EntityID, int) or \
            EntityID > len(hFile['Entity']) or \
            EntityID < 1 or \
            hFile['Entity'][EntityID-1]['EntityType'] != 'Segment':
        ns_RESULT = 'ns_BADENTITY'
        return ns_RESULT, nsSegmentSourceInfo

    # Check SourceID
    if SourceID != 1:
        ns_RESULT = 'ns_BADSOURCE'
        return ns_RESULT, nsSegmentSourceInfo

    ns_RESULT = 'ns_OK'
    nsSegmentSourceInfo['MinVal'] = None
    nsSegmentSourceInfo['MaxVal'] = None
    nsSegmentSourceInfo['Resolution'] = hFile['Entity'][EntityID-1]['Scale']
    nsSegmentSourceInfo['SubSampleShift'] = None
    nsSegmentSourceInfo['LocationX'] = None
    nsSegmentSourceInfo['LocationY'] = None
    nsSegmentSourceInfo['LocationZ'] = None
    nsSegmentSourceInfo['LocationUser'] = None
    nsSegmentSourceInfo['HighFreqCorner'] = None
    nsSegmentSourceInfo['HighFreqOrder'] = None
    nsSegmentSourceInfo['HighFilterType'] = None
    nsSegmentSourceInfo['LowFreqCorner'] = None
    nsSegmentSourceInfo['LowFreqOrder'] = None
    nsSegmentSourceInfo['LowFilterType'] = None
    nsSegmentSourceInfo['ProbeInfo'] = str(hFile['Entity'][EntityID-1]['ElectrodeID'])

    wantedElectrode = hFile['Entity'][EntityID-1]['ElectrodeID']
    # Get file handle for this entity's nev file
    fid = hFile['FileInfo'][hFile['Entity'][EntityID-1]['FileType']-1]['FileID']
    # Find the NEUEVFLT header that has the info for this electrode
    # NOTE: This method was more or less copied from ns_OpenFile. 
    # Should be improved if it proves to be slow.
    fid.seek(332)
    nExtendedHeaders = np.fromfile(fid, dtype='uint32', count=1)[0]

    # Get list of extended header PacketIDs: (there are 9 different nev
    # extended headers. Each have a 8 char array PacketID and each
    # extended header has 24 bytes of information with a total size of
    # 8+24=32 bytes. 
    PacketIDs = np.fromfile(fid, dtype='S8', count=nExtendedHeaders, offset=8+24).astype('U8')
          
    # Set filter types
    FilterType = ['none', 'Butterworth', 'Chebyshev']
          
    # Get Index of NEUEVWAV extended headers.
    idxEVFLT = np.where(PacketIDs == 'NEUEVFLT')[0]

    for j in idxEVFLT:
        # Seek to the next NEUEVWAV extended header from the beginning
        # of the file. 
        fid.seek(344+(j*32))
        elecID = np.fromfile(fid, dtype='uint16', count=1)[0]
        if elecID == wantedElectrode:
            nsSegmentSourceInfo['HighFreqCorner'] = np.fromfile(fid, dtype='uint32', count=1)[0]
            nsSegmentSourceInfo['HighFreqOrder'] = np.fromfile(fid, dtype='uint32', count=1)[0]
            highType = np.fromfile(fid, dtype='uint16', count=1)[0]
            nsSegmentSourceInfo['HighFilterType'] = FilterType[highType]
            nsSegmentSourceInfo['LowFreqCorner'] = np.fromfile(fid, dtype='uint32', count=1)[0]
            nsSegmentSourceInfo['LowFreqOrder'] = np.fromfile(fid, dtype='uint32', count=1)[0]
            lowType = np.fromfile(fid, dtype='uint16', count=1)[0]
            nsSegmentSourceInfo['LowFilterType'] = FilterType[lowType]
            ns_RESULT = 'ns_OK'
            return ns_RESULT, nsSegmentSourceInfo

    ns_RESULT = 'ns_BADENTITY'
    return ns_RESULT, nsSegmentSourceInfo