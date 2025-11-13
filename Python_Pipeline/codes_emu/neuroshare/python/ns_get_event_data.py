def ns_GetEventData(hFile, EntityID, Index):
    # Initialize output variables
    ns_RESULT = ''
    TimeStamp = []
    Data = []
    DataSize = []

    # Check if hFile is a struct
    if not isinstance(hFile, dict):
        ns_RESULT = 'ns_BADFILE'
        return ns_RESULT, TimeStamp, Data, DataSize

    # Check Entity
    if not isinstance(EntityID, int) or EntityID > len(hFile['Entity']) or EntityID < 1 or hFile['Entity'][EntityID-1]['EntityType'] != 'Event':
        ns_RESULT = 'ns_BADENTITY'
        return ns_RESULT, TimeStamp, Data, DataSize

    # Check Index
    if Index < 1 or Index > hFile['Entity'][EntityID-1]['Count']:
        ns_RESULT = 'ns_BADINDEX'
        return ns_RESULT, TimeStamp, Data, DataSize

    ns_RESULT = 'ns_OK'
    DataSize = 2

    # Construct the fileInfo structure for the specified entityID
    fileInfo = hFile['FileInfo'][hFile['Entity'][EntityID-1]['FileType']-1]

    # Use this list to match the reason for this digital event to be stored.
    packetReason = ['Parallel Input', 'SMA 1', 'SMA 2', 'SMA 3', 'SMA 4', 'Output Echo', 'Digital Input', 'Input Ch 1', 'Input Ch 2', 'Input Ch 3', 'Input Ch 4', 'Input Ch 5']

    # Using the list, get the bit that determines the channel crossing for this entity to be stored.
    idx = packetReason.index(hFile['Entity'][EntityID-1]['Reason'])

    # Adjust index for the old packet reasons.
    if idx > 5:
        idx = idx - 6

    # Create a list of all the digital events with the classification for this digital entity.
    # This means, packetid==0 and the stored Class has an up bit in position idx.
    posEvent = [i for i in range(len(fileInfo['MemoryMap']['Data']['Class'])) if fileInfo['MemoryMap']['Data']['Class'][i] & (1 << idx) and fileInfo['MemoryMap']['Data']['PacketID'][i] == 0]

    # The desired entity is at the end of the above list.
    pos = posEvent[Index-1]

    # Get the timestamp for this event. Use a constant 30kHz sampling rate which should be consistent for all nev files.
    TimeStamp = fileInfo['MemoryMap']['Data']['TimeStamp'][pos] / 30000

    # Calculate the offset in bytes to the data that we are interested in from the start of the nev file.
    offset = fileInfo['BytesHeaders'] + fileInfo['BytesDataPacket'] * pos + 8 + (idx-1)*2

    # Skip to the calculated offset.
    fileInfo['FileID'].seek(offset)

    # If the wanted data is the digital parallel port only unsigned ints are supported, otherwise expect signed ints.
    if idx == 0:
        Data = int.from_bytes(fileInfo['FileID'].read(2), byteorder='little', signed=False)
    else:
        Data = int.from_bytes(fileInfo['FileID'].read(2), byteorder='little', signed=True)

    DataSize = len(Data).to_bytes(2, byteorder='little')

    return ns_RESULT, TimeStamp, Data, DataSize