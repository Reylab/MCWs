import numpy as np

def ns_GetAnalogInfo(hFile, EntityID):
    nsAnalogInfo = {}

    # check hFile
    if not isinstance(hFile, dict):
        ns_RESULT = 'ns_BADFILE'
        return ns_RESULT, nsAnalogInfo

    # check Entity
    if not isinstance(EntityID, int) or EntityID != int(EntityID) or hFile['Entity'][EntityID]['EntityType'] != 'Analog':
        ns_RESULT = 'ns_BADENTITY'
        return ns_RESULT, nsAnalogInfo

    ns_RESULT = 'ns_OK'

    # create fileInfo structure for specified Entity
    fileInfo = hFile['FileInfo'][hFile['Entity'][EntityID]['FileType']-1]

    # load default values of nsAnalogInfo structure
    nsAnalogInfo['SampleRate'] = 30000 / fileInfo['Period']
    nsAnalogInfo['MinVal'] = None
    nsAnalogInfo['MaxVal'] = None
    nsAnalogInfo['Units'] = hFile['Entity'][EntityID]['Units']
    nsAnalogInfo['Resolution'] = hFile['Entity'][EntityID]['Scale']
    nsAnalogInfo['LocationX'] = None
    nsAnalogInfo['LocationY'] = None
    nsAnalogInfo['LocationZ'] = None
    nsAnalogInfo['LocationUser'] = None
    nsAnalogInfo['HighFreqCorner'] = None
    nsAnalogInfo['HighFreqOrder'] = None
    nsAnalogInfo['HighFilterType'] = None
    nsAnalogInfo['LowFreqCorner'] = None
    nsAnalogInfo['LowFreqOrder'] = None
    nsAnalogInfo['LowFilterType'] = None
    nsAnalogInfo['ProbeInfo'] = None

    # if nsx2.2 file load scale and filter information
    if fileInfo['FileTypeID'] == 'NEURALCD' or fileInfo['FileTypeID'] == 'NEUCDFLT':
        FilterType = ['none', 'Butterworth', 'Chebyshev']
        # find index of Entity Channel in Electrode List in order to obtain
        # specific extended header information
        chanIDX = np.where(fileInfo['ElectrodeList'] == hFile['Entity'][EntityID]['ElectrodeID'])[0][0]
        # calculate the start of the CC extended header
        headerPos = 314 + 66 * (chanIDX)
        # skip to ElectrodeLabel
        fileInfo['FileID'].seek(headerPos + 4, 0)
        nsAnalogInfo['ProbeInfo'] = fileInfo['FileID'].read(16).decode('utf-8').strip('\x00 ')
        # skip: Physical Connector, Connector Pin, Min/Max Digital Value
        fileInfo['FileID'].seek(6, 1)
        minMaxAnalog = np.fromfile(fileInfo['FileID'], dtype=np.int16, count=2)
        nsAnalogInfo['MinVal'] = minMaxAnalog[0]
        nsAnalogInfo['MaxVal'] = minMaxAnalog[1]
        # skip: Units
        fileInfo['FileID'].seek(16, 1)
        highFilter = np.fromfile(fileInfo['FileID'], dtype=np.uint32, count=2)
        nsAnalogInfo['HighFreqCorner'] = highFilter[0]
        nsAnalogInfo['HighFreqOrder'] = highFilter[1]
        filterIDX = np.fromfile(fileInfo['FileID'], dtype=np.uint16, count=1)[0]
        nsAnalogInfo['HighFilterType'] = FilterType[filterIDX]
        lowFilter = np.fromfile(fileInfo['FileID'], dtype=np.uint32, count=2)
        nsAnalogInfo['LowFreqCorner'] = lowFilter[0]
        nsAnalogInfo['LowFreqOrder'] = lowFilter[1]
        filterIDX = np.fromfile(fileInfo['FileID'], dtype=np.uint16, count=1)[0]
        nsAnalogInfo['LowFilterType'] = FilterType[filterIDX]

    return ns_RESULT, nsAnalogInfo