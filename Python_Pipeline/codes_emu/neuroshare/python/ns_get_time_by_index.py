import numpy as np

def ns_GetTimeByIndex(hFile, EntityID, Index):
    # Initialize output variables
    ns_RESULT = ''
    Time = []

    # Check hFile
    if not isinstance(hFile, dict):
        ns_RESULT = 'ns_BADFILE'
        return ns_RESULT, Time

    # Check Entity
    if not isinstance(EntityID, int) or \
            EntityID > len(hFile['Entity']) or \
            EntityID < 1 or \
            int(EntityID) != EntityID:
        ns_RESULT = 'ns_BADENTITY'
        return ns_RESULT, Time

    # Check Index
    if not isinstance(Index, int) or \
            Index < 1 or \
            Index > hFile['Entity'][EntityID-1]['Count']:
        ns_RESULT = 'ns_BADINDEX'
        return ns_RESULT, Time

    ns_RESULT = 'ns_OK'
    fileInfo = hFile['FileInfo'][hFile['Entity'][EntityID-1]['FileType']-1]

    if hFile['Entity'][EntityID-1]['EntityType'] == 'Analog':
        nPointsAll = np.cumsum(fileInfo['TimeStamps'][1,:])
        idx = np.where(Index <= nPointsAll)[0][0]
        IndexList = np.concatenate(([0], nPointsAll[:-1])) + 1
        Time = (fileInfo['TimeStamps'][0,idx] + Index - IndexList[idx]) * fileInfo['Period'] / 30000
        return ns_RESULT, Time
    elif hFile['Entity'][EntityID-1]['EntityType'] == 'Segment':
        TimeStamps = fileInfo['MemoryMap']['Data']['TimeStamp'][fileInfo['MemoryMap']['Data']['PacketID'] == hFile['Entity'][EntityID-1]['ElectrodeID']]
        Time = float(TimeStamps[Index-1]) / 30000
        return ns_RESULT, Time
    elif hFile['Entity'][EntityID-1]['EntityType'] == 'Event':
        packetReason = ['Digital Input', 'Input Ch 1', 'Input Ch 2', 'Input Ch 3', 'Input Ch 4', 'Input Ch 5']
        idx = packetReason.index(hFile['Entity'][EntityID-1]['Reason'])
        posEvent = np.where((np.bitwise_and(fileInfo['MemoryMap']['Data']['Class'], 2**idx) == 2**idx) & (fileInfo['MemoryMap']['Data']['PacketID'] == 0))[0]
        Time = float(fileInfo['MemoryMap']['Data']['TimeStamp'][posEvent[Index-1]]) / 30000
        return ns_RESULT, Time