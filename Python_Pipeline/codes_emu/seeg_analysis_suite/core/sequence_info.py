# This class is store data from on sequence which 
# is usually signals received while showing 60 images

class SequenceInfo:
    def __init__(self) -> None:
        self.seq_no = 0
        self.data = []
        self.begin_timestamp = 0
        self.end_timestamp = 0
        self.img_timestamps = [] # Should be an array of 60 values. One for each image
