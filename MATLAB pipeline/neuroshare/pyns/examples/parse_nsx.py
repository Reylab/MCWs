
from tkinter import filedialog

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nsfile import NSFile
from nsentity import EntityType


if __name__ == '__main__':
    # Open file dialog to select a .n* file if one is not specified
    filename = filedialog.askopenfilename(title='Nev/NSX Loader')
    ns_file = NSFile(filename)
    ns_file_info = ns_file.get_file_info()

    print(f'File info: {ns_file_info}')