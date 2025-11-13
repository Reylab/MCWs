def ns_GetNeuralData(hFile, EntityID, StartIndex=None, IndexCount=None):
    # Initialize output variables
    ns_RESULT = ''
    Data = []

    # Check hFile
    if not isinstance(hFile, dict):
        ns_RESULT = 'ns_BADFILE'
        return ns_RESULT, Data

    # Check Entity
    if not isinstance(EntityID, int) or \
            EntityID > len(hFile['Entity']) or \
            EntityID < 1 or \
            hFile['Entity'][EntityID-1]['EntityType'] != 'Neural':
        ns_RESULT = 'ns_BADENTITY'
        return ns_RESULT, Data

    # The neural item count will be useful for checking the validity
    # of StartIndex and IndexCount
    itemCount = hFile['Entity'][EntityID-1]['Count']

    if StartIndex is not None:
        if StartIndex < 1 or StartIndex > hFile['Entity'][EntityID-1]['Count']:
            ns_RESULT = 'ns_BADINDEX'
            return ns_RESULT, Data
    else:
        # If StartIndex is not specified, start at the beginning of the data
        StartIndex = 1

    if IndexCount is not None:
        if IndexCount > itemCount:
            ns_RESULT = 'ns_BADINDEX'
            return ns_RESULT, Data
        endBin = StartIndex + IndexCount - 1
    else:
        # If IndexCount is not specified, go to the end of the neural entities
        endBin = itemCount

    ns_RESULT = 'ns_OK'

    # Get needed variables out of the hFile dict
    elecID = hFile['Entity'][EntityID-1]['ElectrodeID']
    fileInfo = hFile['FileInfo'][hFile['Entity'][EntityID-1]['FileType']]
    class_ = hFile['Entity'][EntityID-1]['Reason']

    # Find instances where we have the wanted electrode id and sorted neural entity
    neuralIndices = (fileInfo['MemoryMap']['Data']['PacketID'] == elecID) & \
                    (fileInfo['MemoryMap']['Data']['Class'] == class_)

    # Get all the timestamps that match our criteria and put them in seconds
    timestamps = fileInfo['MemoryMap']['Data']['TimeStamp'][neuralIndices] / 30000

    # Slice the wanted part of the timestamps
    Data = timestamps[StartIndex-1:endBin-1]

    return ns_RESULT, Data