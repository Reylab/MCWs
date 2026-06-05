def ns_GetNeuralInfo(hFile, EntityID):
    # Initialize output variables
    ns_RESULT = ''
    nsNeuralInfo = {}

    # Check hFile
    if not isinstance(hFile, dict):
        ns_RESULT = 'ns_BADFILE'
        return ns_RESULT, nsNeuralInfo

    # Check Entity
    if not isinstance(EntityID, int) or \
            EntityID > len(hFile['Entity']) or \
            EntityID < 1 or \
            hFile['Entity'][EntityID-1]['EntityType'] != 'Neural':
        ns_RESULT = 'ns_BADENTITY'
        return ns_RESULT, nsNeuralInfo

    ns_RESULT = 'ns_OK'
    entityID = hFile['Entity'][EntityID-1]['ElectrodeID']
    class_ = hFile['Entity'][EntityID-1]['Reason']
    nsNeuralInfo['SourceEntityID'] = entityID
    nsNeuralInfo['SourceUnitID'] = class_
    nsNeuralInfo['ProbeInfo'] = f'module {class_}, pin {entityID}'

    return ns_RESULT, nsNeuralInfo