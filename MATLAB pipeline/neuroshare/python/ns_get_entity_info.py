def ns_GetEntityInfo(hFile, EntityID):
    # check hFile
    if not isinstance(hFile, dict):
        ns_RESULT = 'ns_BADFILE'
        return ns_RESULT, []

    # check Entity
    if EntityID > len(hFile['Entity']) or \
        EntityID < 0 :
        ns_RESULT = 'ns_BADENTITY'
        return ns_RESULT, []

    # list of entity types
    EntityTypes = ['Event', 'Analog', 'Segment', 'Neural']
    ns_RESULT = 'ns_OK'

    # start with empty label
    nsEntityInfo = {'EntityLabel': '', 'EntityType': 0, 'ItemCount': 0}

    # if the Label field on the hFile Entity exists then copy it into the label on the nsEntityInfo
    if 'Label' in hFile['Entity'][EntityID]:
        nsEntityInfo['EntityLabel'] = hFile['Entity'][EntityID]['Label']
        # IJM Added if logic below.
        if len(nsEntityInfo['EntityLabel']) == 1:
            nsEntityInfo['EntityLabel'] = ''

    # if the label on the nsEntityInfo is empty, auto-generate a label
    if not nsEntityInfo['EntityLabel']:
        if hFile['Entity'][EntityID]['EntityType'] == 'Analog':
            file_index = hFile['Entity'][EntityID]['FileType']
            sample_rate = 30 / hFile['FileInfo'][file_index]['Period']
            nsEntityInfo['EntityLabel'] = f'{hFile["Entity"][EntityID]["ElectrodeID"]} - {sample_rate:.0f} kS/s'
        elif hFile['Entity'][EntityID]['EntityType'] == 'Event':
            # nsEntityInfo.EntityLabel = 'digin';
            nsEntityInfo['EntityLabel'] = hFile['Entity'][EntityID]['Reason']
        elif hFile['Entity'][EntityID]['ElectrodeID']:
            nsEntityInfo['EntityLabel'] = f'elec{hFile["Entity"][EntityID]["ElectrodeID"]}'
        else:
            nsEntityInfo['EntityLabel'] = hFile['Entity'][EntityID]['Reason']

    nsEntityInfo['EntityType'] = EntityTypes.index(hFile['Entity'][EntityID]['EntityType'])
    nsEntityInfo['ItemCount'] = hFile['Entity'][EntityID]['Count']

    return ns_RESULT, nsEntityInfo