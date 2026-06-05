def ns_GetSegmentInfo(hFile, EntityID):
    # Initialize output variables
    ns_RESULT = ''
    nsSegmentInfo = {}

    # Check hFile
    if not isinstance(hFile, dict):
        ns_RESULT = 'ns_BADFILE'
        return ns_RESULT, nsSegmentInfo

    # Check Entity
    if not isinstance(EntityID, int) or \
            EntityID > len(hFile['Entity']) or \
            EntityID < 1 or \
            hFile['Entity'][EntityID-1]['EntityType'] != 'Segment':
        ns_RESULT = 'ns_BADENTITY'
        return ns_RESULT, nsSegmentInfo

    ns_RESULT = 'ns_OK'
    nsSegmentInfo['SourceCount'] = 1
    SampWaveforms = (hFile['FileInfo'][hFile['Entity'][EntityID-1]['FileType']-1]['BytesDataPacket']-8) / 2
    nsSegmentInfo['MinSampleCount'] = SampWaveforms
    nsSegmentInfo['MaxSampleCount'] = SampWaveforms
    nsSegmentInfo['SampleRate'] = 30000
    nsSegmentInfo['Units'] = hFile['Entity'][EntityID-1]['Units']

    return ns_RESULT, nsSegmentInfo