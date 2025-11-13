import numpy as np

def ns_GetSegmentData(hFile, EntityID, Index):
    # Initialize output variables
    ns_RESULT = ''
    TimeStamp = []
    Data = []
    SampleCount = []
    UnitID = []

    # Check hFile
    if not isinstance(hFile, dict):
        ns_RESULT = 'ns_BADFILE'
        return ns_RESULT, TimeStamp, Data, SampleCount, UnitID

    # Check Entity
    if not isinstance(EntityID, int) or \
            EntityID > len(hFile['Entity']) or \
            EntityID < 1 or \
            hFile['Entity'][EntityID-1]['EntityType'] != 'Segment':
        ns_RESULT = 'ns_BADENTITY'
        return ns_RESULT, TimeStamp, Data, SampleCount, UnitID

    # Check Index
    if Index < 1 or Index > hFile['Entity'][EntityID-1]['Count']:
        ns_RESULT = 'ns_BADINDEX'
        return ns_RESULT, TimeStamp, Data, SampleCount, UnitID

    ns_RESULT = 'ns_OK'

    # Create fileInfo Structure
    fileInfo = hFile['FileInfo'][hFile['Entity'][EntityID-1]['FileType']-1]
    SampleCount = int((fileInfo['BytesDataPacket'] - 8) / 2)
    PacketIndex = np.where(fileInfo['MemoryMap']['Data']['PacketID'] == hFile['Entity'][EntityID-1]['ElectrodeID'])[0][Index-1]
    TimeStamp = float(fileInfo['MemoryMap']['Data']['TimeStamp'][PacketIndex]) / 30000
    UnitID = fileInfo['MemoryMap']['Data']['Class'][PacketIndex]
    offset = int(fileInfo['BytesHeaders']) + 8 + int(fileInfo['BytesDataPacket']) * int(PacketIndex)

    # Skip to event wave data
    fileInfo['FileID'].seek(offset)
    Data = np.fromfile(fileInfo['FileID'], dtype='int16', count=SampleCount).astype('float64')

    # Scale the data
    Data *= hFile['Entity'][EntityID-1]['Scale']

    return ns_RESULT, TimeStamp, Data, SampleCount, UnitID