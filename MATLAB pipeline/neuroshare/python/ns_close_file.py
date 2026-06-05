def ns_CloseFile(hFile):
    # Ensure hFile looks like file handle returned from ns_OpenFile
    # sometimes, hFile get set to a scalar or [] and this function fails.
    if not isinstance(hFile, dict):
        ns_RESULT = 'ns_BADFILE'
        return ns_RESULT

    ns_RESULT = 'ns_OK'

    fids = [f['FileID'] for f in hFile['FileInfo']]
    errCount = 0
    for fid in fids:
        try:
            fid.close()
        except:
            errCount += 1

    if errCount:
        ns_RESULT = 'ns_BADFILE'

    return ns_RESULT