def ns_GetEventInfo(hFile, EntityID):
    # Initialize output variables
    ns_RESULT = ''
    nsEventInfo = {}

    # Check if hFile is a struct
    if not isinstance(hFile, dict):
        ns_RESULT = 'ns_BADFILE'
        return ns_RESULT, nsEventInfo

    # Check Entity
    if not isinstance(EntityID, int) or \
            EntityID > len(hFile['Entity']) or \
            EntityID < 1 or \
            hFile['Entity'][EntityID-1]['EntityType'] != 'Event':
        ns_RESULT = 'ns_BADENTITY'
        return ns_RESULT, nsEventInfo

    ns_RESULT = 'ns_OK'
    nsEventInfo['EventType'] = 3
    nsEventInfo['MinDataLength'] = 2
    nsEventInfo['MaxDataLength'] = 2
    nsEventInfo['CSVDesc'] = []

    return ns_RESULT, nsEventInfo