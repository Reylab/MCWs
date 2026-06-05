import numpy as np

def ns_GetAnalogDataBlock(hFile, EntityIDs, StartIndex, IndexCount, scaleFlag='scale'):
    ContCount = []
    Data = []
    StartIndex = int(StartIndex)
    IndexCount = int(IndexCount)

    # check input arguments
    if not isinstance(hFile, dict):
        ns_RESULT = 'ns_BADFILE'
        return ns_RESULT, Data

    # check EntityIDs
    if not isinstance(EntityIDs, list) or not all([isinstance(eid, int) for eid in EntityIDs]) or \
            not all([hFile['Entity'][eid]['EntityType'] == 'Analog' for eid in EntityIDs]):
        ns_RESULT = 'ns_BADENTITY'
        return ns_RESULT, Data

    ns_RESULT = 'ns_OK'

    fileType = np.unique([hFile['Entity'][eid]['FileType']-1 for eid in EntityIDs])
    # Require only entities from only a single NSX file at a time
    if len(fileType) > 1:
        ns_RESULT = 'ns_BADENTITY'
        return ns_RESULT, Data
    # create a space to place retrieved data
    fileData = []

    useScale = True
    if scaleFlag == 'unscale':
        useScale = False

    fileInfo = hFile['FileInfo'][fileType[0]]
    # setup bytes per point based on file extension
    if fileInfo['FileTypeID'] == 'NEUCDFLT':
        bytesPerPoint = 4
    else:
        bytesPerPoint = 2
    # Get an entity that corresponds to this entity so that we may make use
    # of Neuroshare functions such as ns_GetIndexByTime below.
    firstEntity = np.where(np.array([hFile['Entity'][eid]['FileType']-1 for eid in EntityIDs]) == fileType[0])[0][0]
    # calculate packet information
    indexTotal = min(StartIndex + IndexCount, hFile['Entity'][firstEntity]['Count'])
    IndexCount = indexTotal - StartIndex
    # Get the number of points for each "pause" or data packet
    nPointAll = np.cumsum(fileInfo['TimeStamps'][0, :])[1]
    startPacket = np.where(StartIndex <= nPointAll)[0][0]
    endPacket = np.where(indexTotal <= nPointAll)[0][0]
    nPacket = endPacket - startPacket
    nChannel = len(fileInfo['ElectrodeList'])
    bytesSkip = bytesPerPoint * nChannel
    if useScale:
        Data = np.zeros((IndexCount, len(EntityIDs)))
    else:
        if bytesPerPoint == 4:
            Data = np.zeros((IndexCount, len(EntityIDs)), dtype=np.float32)
        else:
            Data = np.zeros((IndexCount, len(EntityIDs)), dtype=np.int16)
    # More or less randomly put read only 10e8 points at once (each point is
    # two bytes).  If this number gets bigger, we have less reads, but risk
    # running out of memory.  Should we actually check memory size before this
    # step?
    maxRead = int(np.floor(10e8 / nChannel))
    # calculate olffset to skip to the start of the first packets.
    startDataLoc = startPacket * 9 * (fileInfo['FileTypeID'] == 'NEURALCD' or fileInfo['FileTypeID'] == 'NEUCDFLT') + 9
    offset = int(fileInfo['BytesHeaders']) + int(startDataLoc) + int(bytesSkip) * (StartIndex)
    entityList = np.where(np.array([entity['FileType']-1 for entity in hFile['Entity']]) == fileType[0])[0]
    wantedChannels = np.zeros(len(EntityIDs), dtype=int)
    for i in range(len(wantedChannels)):
        wantedChannels[i] = np.where(entityList == EntityIDs[i])[0][0]
    # wantedChannels = wantedChannels[wantedChannels != 0]
    fileInfo['FileID'].seek(offset, 0)
    # create read size for each data Packet
    # readCount = min(IndexCount, maxRead)
    if nPacket == 0:
        packetSize = [IndexCount]
    else:
        # NOTE: Why not use "nPacket"
        packetSize = fileInfo['TimeStamps'][1, startPacket:endPacket]
        packetSize[0] = packetSize[0] - StartIndex
        packetSize[-1] = packetSize[-1] - (np.sum(packetSize) - IndexCount)
    CountList = np.concatenate(([0], np.cumsum(packetSize)))

    # Read through data taking care with pauses.
    for k in range(nPacket+1):
        currentPoint = CountList[k]
        packetRead = 0
        while packetRead < packetSize[k]:
            # we either read the max number of points or to the end of this set of
            # data packets.
            currentRead = min(packetSize[k] - packetRead, maxRead)
            # read block of data
            if bytesPerPoint == 4:
                readData = np.fromfile(fileInfo['FileID'], dtype=np.float32, count=nChannel * currentRead).reshape((nChannel, currentRead), order='F')
            else:
                readData = np.fromfile(fileInfo['FileID'], dtype=np.int16, count=nChannel * currentRead).reshape((nChannel, currentRead), order='F')
            # Copy wanted data return value.  If scaling will be used, convert to
            # double.
            if useScale:
                Data[currentPoint:currentPoint + currentRead, :] = readData[wantedChannels, :].T
            else:
                Data[currentPoint:currentPoint + currentRead, :] = readData[wantedChannels, :].T
            # advance the read marker for overall file
            currentPoint = currentPoint + currentRead
            # advance the read marker at the packet level
            packetRead = packetRead + currentRead
        fileInfo['FileID'].seek(9, 1)

    return ns_RESULT, Data