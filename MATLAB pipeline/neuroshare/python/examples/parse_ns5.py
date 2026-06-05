# Author: Sunil Mathew
# Date: 18 October 2023
# Example to test python version of neuroshare MATLAB library to read .n* files

import sys
import os

# Add the path to the neuroshare python library to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ns_open_file import ns_OpenFile
from ns_get_entity_info import ns_GetEntityInfo
from ns_get_analog_info import ns_GetAnalogInfo
from ns_close_file import ns_CloseFile
from ns_get_analog_data import ns_GetAnalogData
from ns_get_analog_data_block import ns_GetAnalogDataBlock

entityID = 106
startIndex = 0
indexCount = 14622360

# Opens file dialog to select a .n* file if one is not specified
ns_RESULT, hFile = ns_OpenFile()

# Extract channel info
ns_RESULT, entityInfo = ns_GetEntityInfo(hFile, entityID)

# Get analog info
ns_RESULT, AnalogInfo = ns_GetAnalogInfo(hFile, entityID)

# Get analog data
ns_RESULT, ContCount, Data = ns_GetAnalogData(hFile, entityID, startIndex, indexCount)

# Get analog data block
ns_RESULT, analogData = ns_GetAnalogDataBlock(hFile, list(range(0,5)), 0, 10, 'unscale')

# Close file
ns_RESULT = ns_CloseFile(hFile)