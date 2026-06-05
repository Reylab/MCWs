% Class to handle paths
% if a path ends with '/.' the dot means, that only that folder will be
% added and not the subfolders

classdef reylab_custompath < handle
	
	properties
        root_path = '';
        added_paths = {};
    end
    methods
        function obj=reylab_custompath(paths2add)
            obj.root_path = fileparts(mfilename('fullpath'));
            obj.added_paths = {obj.root_path};
            if exist('paths2add', 'var')
                obj.add(paths2add, 0);
            end
        end
        
        function add(obj, paths2add, abs_path)
            if ~exist('abs_path','var')
                abs_path = false;
            end
            if isempty(paths2add)
                return
            end
            if ~iscell(paths2add)
                paths2add = {paths2add};
            end
            if abs_path
                fullpaths = strcat(replace(paths2add, {'\','/','\\'}, ...
                                {filesep,filesep,filesep}));
            else
                fullpaths = strcat([obj.root_path filesep ],replace(paths2add, ...
                                {'\','/','\\'}, {filesep,filesep,filesep}));
            end
            
            
            addsubfolders = cellfun(@(x) strcmp(x(end-1:end),[filesep '.']),fullpaths,'UniformOutput',true);
            new_paths = cellfun(@(x) genpath(x),fullpaths(~addsubfolders),'UniformOutput',false);
           
            new_paths_simple = cellfun(@(x) x(1:end-2),fullpaths(addsubfolders),'UniformOutput',false);
            new_paths = [new_paths, new_paths_simple];
            cellfun(@(x) addpath(x),new_paths);
            obj.added_paths = [new_paths, obj.added_paths];
            
        end
        
        function rm(obj)
            cellfun(@(x) rmpath(x), obj.added_paths);
        end
    end
end