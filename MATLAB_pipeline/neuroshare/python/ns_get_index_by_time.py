import numpy as np
from ns_get_entity_info import ns_GetEntityInfo

def ns_GetIndexByTime(hFile, EntityID, Time, Flag):
    # Initialize output variables
    ns_RESULT = ''
    Index = []

    # Check if hFile is a struct
    if not isinstance(hFile, dict):
        ns_RESULT = 'ns_BADFILE'
        return ns_RESULT, Index

    # Check Entity
    if not isinstance(EntityID, int) or \
            EntityID > len(hFile['Entity']) or \
            EntityID < 1 or \
            EntityID != EntityID & 0xFFFF:
        ns_RESULT = 'ns_BADENTITY'
        return ns_RESULT, Index

    fileInfo = hFile['FileInfo'][hFile['Entity'][EntityID-1]['FileType']]

    if Time < 0 or Time > fileInfo['TimeSpan']:
        ns_RESULT = 'ns_BADTIME'
        return ns_RESULT, Index

    validFlags = [-1, 0, 1]
    if Flag not in validFlags:
        ns_RESULT = 'ns_BADFLAG'
        return ns_RESULT, Index

    ns_RESULT, EntityInfo = ns_GetEntityInfo(hFile, EntityID)

    # TimeSamp is in units of NIP clock (30kHz) however, is a double and may be non-integer. Must
    # be careful in the below source to return only integer indices.
    TimeSamp = Time * 30000 / fileInfo['Period']

    if hFile['Entity'][EntityID-1]['EntityType'] == 'Analog':
        TimeList = np.vstack((fileInfo['TimeStamps'][0], np.cumsum(fileInfo['TimeStamps'][1::2])))
        idx = np.argmax(TimeList[1]+1-TimeSamp > 0)
        IndexList = np.hstack((0, np.cumsum(fileInfo['TimeStamps'][2::2])+1))
        Index = IndexList[idx] + max(int(np.floor(TimeSamp)) - TimeList[0, idx], 0)
        Index = min(Index, EntityInfo['ItemCount'])
    elif hFile['Entity'][EntityID-1]['EntityType'] == 'Segment':
        TimeStamps = fileInfo['MemoryMap']['Data']['TimeStamp'][fileInfo['MemoryMap']['Data']['PacketID'] == hFile['Entity'][EntityID-1]['ElectrodeID']]
        Index = np.argmin(np.abs(TimeStamps.astype(float)-TimeSamp))
        if Flag and (TimeStamps[Index] != TimeSamp):
            Index = max(Index+Flag, 1)
            Index = min(Index, EntityInfo['ItemCount'])
    elif hFile['Entity'][EntityID-1]['EntityType'] == 'Event':
        packetReason = ['Parallel Input', 'SMA 1', 'SMA 2', 'SMA 3', 'SMA 4', 'Output Echo', 'Digital Input', 'Input Ch 1', 'Input Ch 2', 'Input Ch 3', 'Input Ch 4', 'Input Ch 5']
        idx = packetReason.index(hFile['Entity'][EntityID-1]['Reason'])
        TimeStamps = fileInfo['MemoryMap']['Data']['TimeStamp'][(np.bitwise_and(fileInfo['MemoryMap']['Data']['Class'], 2**idx) == 2**idx) & (fileInfo['MemoryMap']['Data']['PacketID'] == 0)]
        Index = np.argmin(np.abs(TimeStamps.astype(float)-TimeSamp))
        if Flag and (TimeStamps[Index] != TimeSamp):
            Index = max(Index+Flag, 1)
            Index = min(Index, EntityInfo['ItemCount'])

    ns_RESULT = 'ns_OK'
    return ns_RESULT, Index