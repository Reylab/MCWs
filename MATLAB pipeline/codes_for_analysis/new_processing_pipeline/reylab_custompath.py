import os
import sys

class reylab_custompath:
    def __init__(self, paths2add=None):
        self.root_path = os.path.dirname(os.path.abspath(__file__))
        self.added_paths = [self.root_path]
        if paths2add:
            self.add(paths2add, False)

    def add(self, paths2add, abs_path=False):
        if not paths2add:
            return
        if not isinstance(paths2add, list):
            paths2add = [paths2add]
        if abs_path:
            fullpaths = [os.path.abspath(path) for path in paths2add]
        else:
            fullpaths = [os.path.join(self.root_path, path) for path in paths2add]

        addsubfolders = [path.endswith('/.') for path in fullpaths]
        new_paths = [os.pathsep.join([path for path in os.walk(fullpath)]) for fullpath, addsub in zip(fullpaths, addsubfolders) if not addsub]
        new_paths_simple = [fullpath[:-2] for fullpath, addsub in zip(fullpaths, addsubfolders) if addsub]
        new_paths = new_paths + new_paths_simple
        [sys.path.append(path) for path in new_paths]
        self.added_paths = new_paths + self.added_paths

    def rm(self):
        [sys.path.remove(path) for path in self.added_paths]