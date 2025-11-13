import struct

def ns_GetFileInfo(hFile):
    # Initialize output variables
    ns_RESULT = ''
    nsFileInfo = []

    # Check if hFile is a struct
    if not isinstance(hFile, dict):
        ns_RESULT = 'ns_BADFILE'
        return ns_RESULT, nsFileInfo

    # Initialize nsFileInfo structure
    ns_RESULT = 'ns_OK'
    nsFileInfo = {'FileType': '', 'EntityCount': 0, 'AppName': '', 'FileComment': '',
                  'TimeSpan': 0, 'TimeStampResolution': 0, 'Time_Year': 0, 'Time_Month': 0,
                  'Time_Day': 0, 'Time_Hour': 0, 'Time_Min': 0, 'Time_Sec': 0, 'Time_MilliSec': 0}

    for fileInfo in hFile['FileInfo']:
        try:
            f = fileInfo['FileID']
            if fileInfo['Type'] == 'nev':
                # Seek to Time Origin header field
                f.seek(28, 0)
                Date = struct.unpack('8H', f.read(16))
                nsFileInfo['Time_Year'] = Date[0]
                nsFileInfo['Time_Month'] = Date[1]
                nsFileInfo['Time_Day'] = Date[3]
                nsFileInfo['Time_Hour'] = Date[4]
                nsFileInfo['Time_Min'] = Date[5]
                nsFileInfo['Time_Sec'] = Date[6]
                nsFileInfo['Time_MilliSec'] = Date[7]
                nsFileInfo['AppName'] = f.read(32).decode().rstrip('\x00')
                # In Ripple files (recorded with Trellis) the comment field is only
                # 252 characters and the NIP timestamp (uint32) for the recording
                # started is packed in the last four bytes.
                if 'Trellis' in nsFileInfo['AppName']:
                    nsFileInfo['FileComment'] = f.read(252).decode().rstrip('\x00')
                    # With Ripple data get the NIP time
                    nsFileInfo['NIPTime'] = struct.unpack('I', f.read(4))[0]
                else:
                    nsFileInfo['FileComment'] = f.read(256).decode().rstrip('\x00')
            else:  # if not nev
                # Check if nsx 2.2 file and that file comment is not yet written
                if not nsFileInfo['FileComment'] and fileInfo['FileTypeID'] == 'NEURALCD':
                    # Seek to file comment
                    f.seek(30, 0)
                    # In Ripple files (recorded with Trellis) the comment field is only
                    # 252 characters and the NIP timestamp (uint32) for the recording
                    # started is packed in the last four bytes.
                    #
                    # We start first read the first 252 chars of the comment
                    nsFileInfo['FileComment'] = f.read(252).decode().rstrip('\x00')
                    if 'Trellis' in nsFileInfo['FileComment']:
                        # With Ripple data get the NIP time
                        nsFileInfo['NIPTime'] = struct.unpack('I', f.read(4))[0]
                        # NSX 2.2 doesn't have an explicit space for app name so
                        # it's packed in the FileComment.  We'll repeat this in the
                        # AppName
                        nsFileInfo['FileComment'] = nsFileInfo['FileComment'].rstrip()
                        nsFileInfo['AppName'] = nsFileInfo['FileComment']
                    else:
                        nsFileInfo['FileComment'] = nsFileInfo['FileComment'] + \
                            f.read(4).decode().rstrip('\x00')
                    # Seek to Time Origin
                    f.seek(8, 0)
                    Date = struct.unpack('8H', f.read(16))
                    nsFileInfo['Time_Year'] = Date[0]
                    nsFileInfo['Time_Month'] = Date[1]
                    nsFileInfo['Time_Day'] = Date[3]
                    nsFileInfo['Time_Hour'] = Date[4]
                    nsFileInfo['Time_Min'] = Date[5]
                    nsFileInfo['Time_Sec'] = Date[6]
                    nsFileInfo['Time_MilliSec'] = Date[7]
        except IOError:
            ns_RESULT = 'ns_FILEERROR'
            return ns_RESULT, nsFileInfo

    nsFileInfo['FileType'] = ', '.join([fileInfo['Type'] for fileInfo in hFile['FileInfo']])
    nsFileInfo['EntityCount'] = len(hFile['Entity'])
    nsFileInfo['TimeSpan'] = hFile['TimeSpan'] / 30000
    nsFileInfo['TimeStampResolution'] = min([fileInfo['Period'] for fileInfo in hFile['FileInfo']]) / 30000

    return ns_RESULT, nsFileInfo