import sys

def ns_GetLastErrorMsg():
    # Initialize output variables
    ns_RESULT = ''
    LastError = ''

    # Get last error message
    try:
        raise Exception
    except:
        LastError = str(sys.exc_info()[1])

    ns_RESULT = 'ns_OK'
    return ns_RESULT, LastError