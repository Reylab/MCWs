import numpy as np

def ns_GetAnalogData(hFile, EntityID, StartIndex, IndexCount, scaleFlag=None):
    ContCount = []
    Data = []

    # check input arguments
    if not isinstance(hFile, dict):
        ns_RESULT = 'ns_BADFILE'
        return ns_RESULT, ContCount, Data

    # check EntityID
    if not isinstance(EntityID, int) or EntityID != int(EntityID) or hFile['Entity'][EntityID]['EntityType'] != 'Analog':
        ns_RESULT = 'ns_BADENTITY'
        return ns_RESULT, ContCount, Data

    # Check IndexCount
    if not isinstance(StartIndex, int) or not isinstance(IndexCount, int) or IndexCount < 1 or StartIndex < 0:
        ns_RESULT = 'ns_BADINDEX'
        return ns_RESULT, ContCount, Data

    ns_RESULT = 'ns_OK'

    # create fileInfo structure
    fileInfo = hFile['FileInfo'][hFile['Entity'][EntityID]['FileType']-1]

    # calculate packet information
    IndexTotal = min(StartIndex + IndexCount, hFile['Entity'][EntityID]['Count'])
    IndexCount = IndexTotal - StartIndex
    Data = np.zeros((IndexCount, 1))
    nPointAll = np.cumsum(fileInfo['TimeStamps'][0][1])[0]
    StartPacket = np.argmax(StartIndex <= nPointAll)
    EndPacket = np.argmax(IndexTotal <= nPointAll)
    nPacket = EndPacket - StartPacket
    # create read size for each data Packet
    if nPacket == 0:
        PacketSize = [IndexCount]
    else:
        PacketSize = fileInfo['TimeStamps'][0][1][StartPacket:EndPacket]
        PacketSize[0] = PacketSize[0] - StartIndex
        PacketSize[-1] = PacketSize[-1] - (np.sum(PacketSize) - IndexCount)
    # calculate offset
    if fileInfo['FileTypeID'] == 'NEUCDFLT':
        bytesPerPoint = 4
    else:
        bytesPerPoint = 2
    bytesSkip = bytesPerPoint * len(fileInfo['ElectrodeList']) - bytesPerPoint
    startDataLoc = (StartPacket * 9 * (fileInfo['FileTypeID'] == 'NEURALCD' or fileInfo['FileTypeID'] == 'NEUCDFLT')) + 9
    offset = fileInfo['BytesHeaders'] + startDataLoc + (bytesSkip + bytesPerPoint) * (StartIndex) + \
             bytesPerPoint * np.argmax(fileInfo['ElectrodeList'] == hFile['Entity'][EntityID]['ElectrodeID']) - bytesPerPoint

    fileInfo['FileID'].seek(offset, 0)

    # read data
    CountList = np.concatenate(([0], np.cumsum(PacketSize)))
    for k in range(nPacket+1):
        if bytesPerPoint == 4:
            Data[CountList[k]:CountList[k + 1]] = np.fromfile(fileInfo['FileID'], dtype=np.float32, count=PacketSize[k])[:,np.newaxis]
        else:
            Data[CountList[k]:CountList[k + 1]] = np.fromfile(fileInfo['FileID'], dtype=np.int16, count=PacketSize[k])[:,np.newaxis]
        fileInfo['FileID'].seek(9, 1)

    # set size of returned data
    ContCount = len(Data)

    # if using float streams the data is already scaled
    if bytesPerPoint == 4:
        return ns_RESULT, ContCount, Data

    # apply scale factor
    if scaleFlag is not None and scaleFlag.lower() == 'unscale':
        return ns_RESULT, ContCount, Data
    Data = Data * hFile['Entity'][EntityID]['Scale']

    return ns_RESULT, ContCount, Data